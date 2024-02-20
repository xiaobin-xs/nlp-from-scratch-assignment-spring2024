import argparse
import re
import sys

import argparse
import os
from requests import Session
from typing import Generator, Union

import urllib3
import requests
import json

S2_API_KEY = os.environ['S2_API_KEY']


def get_paper(session: Session, paper_id: str, fields: str = 'paperId,title', **kwargs) -> dict:
    params = {
        'fields': fields,
        **kwargs,
    }
    headers = {
        'X-API-KEY': S2_API_KEY,
    }

    with session.get(f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}', params=params, headers=headers) as response:
        response.raise_for_status()
        return response.json()


def download_pdf(session: Session, url: str, path: str, user_agent: str = 'requests/2.0.0'):
    # send a user-agent to avoid server error
    headers = {
        'user-agent': user_agent,
    }

    # stream the response to avoid downloading the entire file into memory
    with session.get(url, headers=headers, stream=True, verify=False) as response:
        # check if the request was successful
        response.raise_for_status()

        if response.headers['content-type'] != 'application/pdf':
            raise Exception('The response is not a pdf')

        with open(path, 'wb') as f:
            # write the response to the file, chunk_size bytes at a time
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def download_paper(session: Session, paper_id: str, directory: str = 'papers', user_agent: str = 'requests/2.0.0') -> Union[str, None]:
    paper = get_paper(session, paper_id, fields='paperId,isOpenAccess,openAccessPdf')

    # check if the paper is open access
    if not paper['isOpenAccess']:
        return None

    if paper['openAccessPdf'] is None:
        return None

    paperId: str = paper['paperId']
    pdf_url: str = paper['openAccessPdf']['url']
    pdf_path = os.path.join(directory, f'{paperId}.pdf')

    # create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # check if the pdf has already been downloaded
    if not os.path.exists(pdf_path):
        download_pdf(session, pdf_url, pdf_path, user_agent=user_agent)

    return pdf_path


def download_papers(paper_ids: list[str], directory: str = 'papers', user_agent: str = 'requests/2.0.0') -> Generator[tuple[str, Union[str, None, Exception]], None, None]:
    # use a session to reuse the same TCP connection
    with Session() as session:
        for paper_id in paper_ids:
            try:
                yield paper_id, download_paper(session, paper_id, directory=directory, user_agent=user_agent)
            except Exception as e:
                yield paper_id, e

def resolve_author(desc: str):
    req_fields = 'authorId,name,url'

    if re.match('\\d+', desc):  # ID given
        rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{desc}',
                           params={'fields': req_fields})
        rsp.raise_for_status()
        return rsp.json()

    else:  # search
        rsp = requests.get('https://api.semanticscholar.org/graph/v1/author/search',
                           params={'query': desc, 'fields': req_fields})
        rsp.raise_for_status()

        results = rsp.json()
        if results['total'] == 0:  # no results
            print(f'Could not find author "{desc}"')
            sys.exit(1)
        elif results['total'] == 1:  # one unambiguous match
            return results['data'][0]
        else:  # multiple matches
            print(f'Multiple authors matched "{desc}".')
            for author in results['data']:
                print(author)
            print('Re-run with a specific ID.')
            sys.exit(1)

# 1700325 3312309 2109449533 144987107 145431806 1700007 145472333 1758714 2260563 47070750 145001267 7661726 7975935
# 1686960 143900005 1706595 49933077 144287919 3407646 1723120 1681921 35959897 1783635 2729164 1890127 2109454942 2268272
# 1724972 1746678 2129663 143977260 144628574 35729970
def get_author_papers(author_id):
    rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers',
                       params={'fields': 'title,url,year', 'limit': 100})
    rsp.raise_for_status()
    papers = rsp.json()['data']
    return [paper for paper in papers if paper['year'] == 2023]


def get_author_id():
    parser = argparse.ArgumentParser()
    parser.add_argument('author')
    args = parser.parse_args()

    author = resolve_author(args.author)
    print(author)
    print(f"Left author: {author['name']} {author['url']}")

    papers = get_author_papers(author['authorId'])
    print(f'Found {len(papers)} left_papers')
    for paper in papers:
        print(paper)
        print(f'{paper["year"]} {paper["title"]} {paper["url"]}')
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', default='../../data/papers')
    parser.add_argument('--user-agent', '-u', default='requests/2.0.0')
    parser.add_argument('author_ids', nargs='+', help='List of author IDs')
    args = parser.parse_args()

    for author_id in args.author_ids:
        author_meta = resolve_author(author_id)
        author_meta_path = os.path.join(args.directory, f"author_{author_meta['name']}_metadata.txt")
        author_papers = get_author_papers(author_id)
        with open(author_meta_path, 'w') as f:
            # Write author metadata
            f.write(json.dumps(author_meta, indent=4))
            f.write('\n\n')
            print(f'Found {len(author_papers)} papers published after 2023 for author {author_meta["name"]}')
            for paper in author_papers:
                f.write(json.dumps(paper, indent=4))
                f.write('\n')

        print(f"Saved metadata for author {author_meta['name']} in {author_meta_path}")
        with Session() as session:
            for paper in author_papers:
                try:
                    download_result = download_paper(session, paper['paperId'], directory=args.directory, user_agent=args.user_agent)
                    if download_result:
                        print(f"Downloaded '{paper['title']}' to '{download_result}'")
                except Exception as e:
                    print(f"Error downloading '{paper['title']}': {e}")


if __name__ == '__main__':
    # get_author_id()
    main()