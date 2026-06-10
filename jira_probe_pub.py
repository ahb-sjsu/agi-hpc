import requests
import pandas as pd
import time

# =========================
# CONFIG
# =========================
JIRA_URL = "<URL>"
API_TOKEN = "<TOKEN>"
PROJECT_KEY = "<KEY>"

OUTPUT_PREFIX = "nst_jira"

# =========================
# SESSION AUTH
# =========================
session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
})

# =========================
# HELPER: API CALL
# =========================
def jira_get(url, params=None):
    response = session.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        response.raise_for_status()

    return response.json()


# =========================
# HELPER: PAGINATION
# =========================
def paginated_get(url, params=None, key="values"):
    results = []
    start_at = 0

    while True:
        if params:
            params["startAt"] = start_at
        else:
            params = {"startAt": start_at}

        data = jira_get(url, params)

        chunk = data.get(key, data.get("issues", []))
        results.extend(chunk)

        total = data.get("total", len(results))
        max_results = data.get("maxResults", len(chunk))

        if start_at + max_results >= total:
            break

        start_at += max_results
        time.sleep(0.2)

    return results


# =========================
# 1. GET ALL FIELDS
# =========================
def get_fields():
    print("Fetching fields...")
    url = f"{JIRA_URL}/rest/api/2/field"
    data = jira_get(url)
    pd.DataFrame(data).to_csv(f"{OUTPUT_PREFIX}_fields.csv", index=False)


# =========================
# 2. GET ALL ISSUES (WITH CHANGELOG)
# =========================
def get_all_issues():
    print("Pulling all issues...")

    issues = []
    start_at = 0
    max_results = 100

    while True:
        url = f"{JIRA_URL}/rest/api/2/search"
        params = {
            "jql": f"project={PROJECT_KEY}",
            "startAt": start_at,
            "maxResults": max_results,
            "expand": "changelog"
        }

        data = jira_get(url, params)
        issues.extend(data["issues"])

        print(f"Fetched {len(issues)} issues...")

        if start_at + max_results >= data["total"]:
            break

        start_at += max_results
        time.sleep(0.2)

    print(f"✅ Total issues: {len(issues)}")
    return issues


# =========================
# 3. GET BOARDS
# =========================
def get_boards():
    print("Fetching boards...")
    url = f"{JIRA_URL}/rest/agile/1.0/board"
    boards = paginated_get(url)
    pd.DataFrame(boards).to_csv(f"{OUTPUT_PREFIX}_boards.csv", index=False)
    return boards


# =========================
# 4. GET SPRINTS
# =========================
def get_sprints(board_id):
    url = f"{JIRA_URL}/rest/agile/1.0/board/{board_id}/sprint"
    return paginated_get(url)


# =========================
# 5. GET SPRINT ISSUES
# =========================
def get_sprint_issues(board_id, sprint_id):
    url = f"{JIRA_URL}/rest/agile/1.0/board/{board_id}/sprint/{sprint_id}/issue"
    return paginated_get(url, key="issues")


# =========================
# 6. FLATTEN ISSUES
# =========================
def flatten_issues(issues):
    rows = []

    print("Flattening issues...")

    for i, issue in enumerate(issues):
        if i % 500 == 0:
            print(f"Processing issue {i}/{len(issues)}")

        fields = issue["fields"]

        status_changes = []
        for history in issue.get("changelog", {}).get("histories", []):
            for item in history["items"]:
                if item["field"] == "status":
                    status_changes.append({
                        "from": item.get("fromString"),
                        "to": item.get("toString"),
                        "date": history.get("created")
                    })

        rows.append({
            "key": issue["key"],
            "summary": fields.get("summary"),
            "issue_type": fields.get("issuetype", {}).get("name"),
            "status": fields.get("status", {}).get("name"),
            "priority": fields.get("priority", {}).get("name"),
            "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
            "created": fields.get("created"),
            "updated": fields.get("updated"),

            # UPDATE AFTER PROBE
            "story_points": fields.get("customfield_10002"),
            "team": fields.get("customfield_XXXXX"),
            "pi": fields.get("customfield_YYYYY"),

            "status_history": status_changes
        })

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():

    get_fields()

    issues = get_all_issues()
    df_issues = flatten_issues(issues)
    df_issues.to_csv(f"{OUTPUT_PREFIX}_issues.csv", index=False)

    boards = get_boards()

    sprint_rows = []
    sprint_issue_rows = []

    print("Fetching sprints and sprint issues...")

    for board in boards:
        board_id = board["id"]

        try:
            sprints = get_sprints(board_id)

            for sprint in sprints:
                sprint_rows.append({
                    "board_id": board_id,
                    "board_name": board["name"],
                    "sprint_id": sprint["id"],
                    "sprint_name": sprint["name"],
                    "state": sprint["state"],
                    "startDate": sprint.get("startDate"),
                    "endDate": sprint.get("endDate")
                })

                issues = get_sprint_issues(board_id, sprint["id"])

                for issue in issues:
                    sprint_issue_rows.append({
                        "sprint_id": sprint["id"],
                        "issue_key": issue["key"]
                    })

        except Exception as e:
            print(f"Skipping board {board_id}: {e}")

    pd.DataFrame(sprint_rows).to_csv(f"{OUTPUT_PREFIX}_sprints.csv", index=False)
    pd.DataFrame(sprint_issue_rows).to_csv(f"{OUTPUT_PREFIX}_sprint_issues.csv", index=False)

    print("✅ Done!")


if __name__ == "__main__":
    main()
