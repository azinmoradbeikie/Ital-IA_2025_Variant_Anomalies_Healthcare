import os
import random
import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe
import networkx as nx
from networkx.algorithms import community
from tqdm import tqdm

import config


def get_trace_profile(case_df, unique_activities, unique_resources, all_possible_transitions):
    activities = case_df['concept:name'].tolist()
    resources = case_df['org:resource'].tolist()

    activity_vec = [1 if act in activities else 0 for act in unique_activities]
    resource_counts = case_df['org:resource'].value_counts()
    resource_vec = [resource_counts.get(res, 0) for res in unique_resources]

    transitions = set(f"{activities[i]}→{activities[i+1]}" for i in range(len(activities) - 1))
    transition_vec = [1 if t in transitions else 0 for t in sorted(all_possible_transitions)]

    return activity_vec + resource_vec + transition_vec


def process_data(seed, df_full, all_case_ids):
    random.seed(seed)
    np.random.seed(seed)

    sampled_case_ids = random.sample(list(all_case_ids), int(config.SAMPLE_RATIO * len(all_case_ids)))
    df_sample = df_full[df_full['case:concept:name'].isin(sampled_case_ids)].copy()

    unique_activities = sorted(df_sample['concept:name'].unique())
    unique_resources = sorted(df_sample['org:resource'].dropna().unique())
    all_possible_transitions = {f"{a1}→{a2}" for a1 in unique_activities for a2 in unique_activities}

    case_profiles = {}
    for case_id, case_df in tqdm(df_sample.groupby('case:concept:name'), desc=f"Processing cases (Seed {seed})"):
        try:
            case_profiles[case_id] = get_trace_profile(case_df, unique_activities, unique_resources, all_possible_transitions)
        except Exception as e:
            print(f"Skipped case {case_id}: {e}")

    case_ids = list(case_profiles.keys())
    X = np.array([case_profiles[case_id] for case_id in case_ids])
    n_cases = len(case_ids)
    G = nx.Graph()

    for i, case_id in enumerate(case_ids):
        activities = df_sample[df_sample['case:concept:name'] == case_id]['concept:name'].tolist()
        resources = df_sample[df_sample['case:concept:name'] == case_id]['org:resource'].tolist()
        G.add_node(i,
                   label=str(case_id),
                   activities=str(activities),
                   resources=str(resources))

    knn_results = []
    for i in tqdm(range(n_cases), desc=f"Building K-NN (Seed {seed})"):
        distances = []
        for j in range(n_cases):
            if i != j:
                dist = levenshtein_distance(str(X[i]), str(X[j]))
                distances.append((j, dist))

        nearest = sorted(distances, key=lambda x: x[1])[:config.K_NEIGHBORS]
        for j, dist in nearest:
            G.add_edge(i, j, weight=1 - dist)

        knn_results.append({
            'case_id': case_ids[i],
            'neighbors': [case_ids[j] for j, _ in nearest],
            'distances': [d for _, d in nearest],
            'avg_distance': sum(d for _, d in nearest) / config.K_NEIGHBORS
        })

    seed_dir = os.path.join(config.OUTPUT_DIR, f"results_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    df_sample.to_csv(f"{seed_dir}/preprocessed_log.csv", index=False)
    pd.DataFrame(knn_results).to_csv(f'{seed_dir}/knn_results.csv', index=False)

    communities = list(community.louvain_communities(G, resolution=1.0))
    threshold = max(5, int(config.MIN_COMMUNITY_THRESHOLD_RATIO * len(case_ids)))
    anomalous_communities = [c for c in communities if len(c) < threshold]

    with open(f"{seed_dir}/community_report.txt", "w") as f:
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Total communities: {len(communities)}\n")
        f.write(f"Anomalous communities (size < {threshold}): {len(anomalous_communities)}\n\n")
        f.write("Example anomalous traces:\n")
        for i, comm in enumerate(anomalous_communities[:3]):
            f.write(f"\nAnomaly Community {i + 1} (Size: {len(comm)}):\n")
            for node in list(comm)[:3]:
                case_id = case_ids[node]
                activities = df_sample[df_sample['case:concept:name'] == case_id]['concept:name'].tolist()
                f.write(f"  Case {case_id}: {' -> '.join(activities)}\n")

    nx.write_gexf(G, f"{seed_dir}/anomalies_graph.gexf")
    return len(anomalous_communities)


if __name__ == "__main__":
    print("Loading event log...")
    log = xes_importer.apply(config.XES_FILE_PATH)
    df_full = convert_to_dataframe(log)
    all_case_ids = df_full['case:concept:name'].unique()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    results = []
    for i in range(config.NUM_RUNS):
        current_seed = random.randint(1, 10000)
        print(f"\n=== Running iteration {i + 1} with seed {current_seed} ===")
        num_anomalies = process_data(current_seed, df_full, all_case_ids)
        results.append({'seed': current_seed, 'anomalies_found': num_anomalies})
        print(f"Completed run {i + 1}. Found {num_anomalies} anomalous communities.")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'all_runs_summary.csv'), index=False)

    print("\n=== All runs completed ===")
    print("Summary of anomalies found:")
    print(summary_df)
