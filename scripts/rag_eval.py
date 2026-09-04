import json
import argparse

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="eval/rag/queries.jsonl")
    args = parser.parse_args()
    
    print(f"Loading queries from {args.queries}")
    with open(args.queries, 'r') as f:
        for line in f:
            query = json.loads(line)
            print(f"Query: {query.get('query')}")
            
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate()
