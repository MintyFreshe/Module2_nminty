import csv
import networkx as nx
import requests


API_URL = "https://en.wikipedia.org/w/api.php"
SEED_PAGE = "Artificial intelligence"
MAX_SEED_LINKS = 35
OUTPUT_GEXF = "ai_wikipedia_network.gexf"
OUTPUT_NODES_CSV = "ai_wikipedia_nodes.csv"
OUTPUT_EDGES_CSV = "ai_wikipedia_edges.csv"


def get_links(title, session):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "links",
        "pllimit": "max",
        "plnamespace": 0,
    }

    links = []

    while True:
        response = session.get(API_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for link in page.get("links", []):
                link_title = link.get("title")
                if link_title:
                    links.append(link_title)

        if "continue" not in data:
            break

        params.update(data["continue"])

    return list(dict.fromkeys(links))


def build_network(seed, max_seed_links=MAX_SEED_LINKS):
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "INST414-Wikipedia-PageRank/1.0 (network visualization)"}
    )

    graph = nx.DiGraph()

    seed_links = get_links(seed, session)[:max_seed_links]
    allowed_nodes = {seed, *seed_links}

    graph.add_node(seed)

    for link in seed_links:
        graph.add_edge(seed, link)

    for page in seed_links:
        outgoing_links = get_links(page, session)
        for target in outgoing_links:
            if target in allowed_nodes:
                graph.add_edge(page, target)

    return graph


def compute_pagerank(graph, damping=0.85, max_iterations=100, tolerance=1e-6):
    nodes = list(graph.nodes())
    if not nodes:
        return {}

    node_count = len(nodes)
    scores = {node: 1.0 / node_count for node in nodes}
    incoming = {node: list(graph.predecessors(node)) for node in nodes}
    outgoing_counts = {node: graph.out_degree(node) for node in nodes}

    for _ in range(max_iterations):
        new_scores = {}
        dangling_total = sum(scores[node] for node in nodes if outgoing_counts[node] == 0)

        for node in nodes:
            rank_sum = 0.0
            for source in incoming[node]:
                rank_sum += scores[source] / outgoing_counts[source]

            new_scores[node] = ((1.0 - damping) / node_count) + damping * (
                rank_sum + dangling_total / node_count
            )

        delta = sum(abs(new_scores[node] - scores[node]) for node in nodes)
        scores = new_scores

        if delta < tolerance:
            break

    return scores


def add_node_attributes(graph, pagerank_scores, seed):
    for node in graph.nodes():
        graph.nodes[node]["label"] = node
        graph.nodes[node]["pagerank"] = pagerank_scores[node]
        graph.nodes[node]["is_seed"] = node == seed


def export_for_gephi(graph):
    nx.write_gexf(graph, OUTPUT_GEXF)

    with open(OUTPUT_NODES_CSV, "w", newline="", encoding="utf-8") as nodes_file:
        writer = csv.writer(nodes_file)
        writer.writerow(["id", "label", "pagerank", "is_seed"])
        for node, attributes in graph.nodes(data=True):
            writer.writerow(
                [
                    node,
                    attributes.get("label", node),
                    attributes.get("pagerank", 0.0),
                    attributes.get("is_seed", False),
                ]
            )

    with open(OUTPUT_EDGES_CSV, "w", newline="", encoding="utf-8") as edges_file:
        writer = csv.writer(edges_file)
        writer.writerow(["source", "target"])
        for source, target in graph.edges():
            writer.writerow([source, target])


def main():
    try:
        graph = build_network(SEED_PAGE)
    except requests.RequestException as exc:
        print(f"Could not build graph: {exc}")
        return

    if graph.number_of_nodes() == 0:
        print("No graph was built.")
        return

    pagerank_scores = compute_pagerank(graph)
    add_node_attributes(graph, pagerank_scores, SEED_PAGE)
    export_for_gephi(graph)

    top_pages = sorted(
        (
            (page, score)
            for page, score in pagerank_scores.items()
            if page != SEED_PAGE
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    print(f"Seed page: {SEED_PAGE}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Saved graph for Gephi to: {OUTPUT_GEXF}")
    print(f"Saved node table to: {OUTPUT_NODES_CSV}")
    print(f"Saved edge table to: {OUTPUT_EDGES_CSV}")
    print("\nTop pages by PageRank:")
    for page, score in top_pages:
        print(f"{page}: {score:.5f}")


if __name__ == "__main__":
    main()
