import csv
import time
import datetime
import sys

import requests
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
OUTPUT   = 'repos.csv'
PAGES    = 10  
PER_PAGE = 100

PAGE_SLEEP = 5

README_SLEEP = 0.5
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    'name', 'stargazers_count', 'forks_count', 'watchers_count',
    'open_issues_count', 'language', 'has_wiki', 'has_projects',
    'readme_length', 'repo_age_days',
]


def _headers():
    h = {'Accept': 'application/vnd.github+json'}
    if TOKEN and TOKEN != 'your_github_personal_access_token':
        h['Authorization'] = f'token {TOKEN}'
    return h


def fetch_readme_length(full_name):
    url  = f'https://api.github.com/repos/{full_name}/readme'
    resp = requests.get(url, headers=_headers(), timeout=10)
    time.sleep(README_SLEEP)
    if resp.ok:
        return resp.json().get('size', 0)
    return 0


def _handle_rate_limit(resp):
    reset_ts = int(resp.headers.get('X-RateLimit-Reset', 0))
    if reset_ts:
        wait = max(0, reset_ts - int(time.time())) + 5
        print(f'\n[!] Rate limit hit. Sleeping {wait}s until reset...')
        time.sleep(wait)
        return True
    return False


def collect_from_api():
    total = 0

    with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for page in range(1, PAGES + 1):
            print(f'  Fetching page {page}/{PAGES} ...', end=' ', flush=True)

            # Try the search request up to 3 times in case of rate limiting
            for attempt in range(3):
                resp = requests.get(
                    'https://api.github.com/search/repositories',
                    headers=_headers(),
                    params={
                        'q'        : 'stars:>10 stars:<50000',
                        'sort'     : 'stars',
                        'order'    : 'desc',
                        'per_page' : PER_PAGE,
                        'page'     : page,
                    },
                    timeout=15,
                )

                if resp.status_code == 401:
                    print('\n[!] Bad token — check your TOKEN value and re-run.')
                    sys.exit(1)

                if resp.status_code in (429, 403):
                    should_retry = _handle_rate_limit(resp)
                    if should_retry and attempt < 2:
                        continue
                    else:
                        print(f'\n[!] Could not recover from rate limit. '
                              f'Saving {total} repos collected so far.')
                        return total

                if resp.status_code != 200:
                    print(f'\n[!] API error {resp.status_code}: {resp.text[:120]}')
                    break

                break  

            if resp.status_code != 200:
                break

            items = resp.json().get('items', [])
            if not items:
                print('no more items.')
                break

            today = datetime.date.today()
            for i, repo in enumerate(items):
                created = repo.get('created_at', '')[:10]  # just the YYYY-MM-DD part
                try:
                    age = (today - datetime.date.fromisoformat(created)).days
                except ValueError:
                    age = 0

                rm_len = fetch_readme_length(repo['full_name'])

                writer.writerow({
                    'name'              : repo['full_name'],
                    'stargazers_count'  : repo.get('stargazers_count', 0),
                    'forks_count'       : repo.get('forks_count', 0),
                    'watchers_count'    : repo.get('watchers_count', 0),
                    'open_issues_count' : repo.get('open_issues_count', 0),
                    'language'          : repo.get('language') or 'Unknown',
                    'has_wiki'          : int(repo.get('has_wiki', False)),
                    'has_projects'      : int(repo.get('has_projects', False)),
                    'readme_length'     : rm_len,
                    'repo_age_days'     : age,
                })
                total += 1

                if (i + 1) % 10 == 0:
                    print(f'{total}...', end=' ', flush=True)

            print(f'page done. Total: {total} repos.')

            if page < PAGES:
                print(f'  Waiting {PAGE_SLEEP}s before next page...')
                time.sleep(PAGE_SLEEP)

    return total


if __name__ == '__main__':
    print('=' * 60)
    print('  GitHub Repository Data Collector')
    print('=' * 60)

    est_mins = int(PAGES * PER_PAGE * README_SLEEP / 60)
    print(f'\nToken detected. Collecting up to {PAGES * PER_PAGE} repos...')
    print(f'Query: stars:>10 stars:<50000')
    print(f'README sleep: {README_SLEEP}s/repo — estimated time: ~{est_mins} mins\n')

    try:
        n = collect_from_api()
        print(f'\nDone — {n} repositories saved to {OUTPUT}')
    except requests.RequestException as e:
        print(f'\n[!] Network error: {e}')
        sys.exit(1)