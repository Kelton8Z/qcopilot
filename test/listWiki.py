import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *


import aiohttp
import asyncio

# Example configuration for feishuConfig
feishuConfig = {
    'tenantAccessToken': 't-g1046daT3JL7GAYDKWGPDNGMEZIEXOLBE5ITBF6C',
    'endpoint': 'https://open.feishu.cn',
}

async def feishu_fetch(method, path, payload):
    authorization = f"Bearer {feishuConfig['tenantAccessToken']}"
    headers = {
        'Authorization': authorization,
        'Content-Type': 'application/json; charset=utf-8',
        'User-Agent': 'feishu-pages',
    }

    url = f"{feishuConfig['endpoint']}{path}"

    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, json=payload, headers=headers) as response:
            if response.status != 200:
                print(f"Failed to fetch from Feishu: {response.status}")
                return None
            response_data = await response.json()

    code = response_data.get('code')
    data = response_data.get('data')
    msg = response_data.get('msg')

    if code != 0:
        print(f"feishuFetch code: {code}, msg: {msg}")
        return None

    return data

async def feishu_fetch_with_iterator(method, path, payload={}):
    page_token = ""
    has_more = True
    results = []

    while has_more:
        data = await feishu_fetch(method, path, {**payload, 'page_token': page_token})

        if data and 'items' in data:
            results.extend(data['items'])
            has_more = data.get('has_more', False)
            page_token = data.get('page_token', "")
        else:
            has_more = False
            page_token = ""

    return results

async def fetch_doc_info(space_id, node_token):
    data = await feishu_fetch('GET', '/open-apis/wiki/v2/spaces/get_node', {
        'token': node_token,
    })

    node = data.get('node')
    if not node:
        print(f"Node not found: {node_token}, data: {data}")
        return None

    return node

async def fetch_all_docs(space_id, depth=0, parent_node_token=None, visited=None):
    if visited is None:
        visited = set()

    if parent_node_token in visited:
        print(f"Node already visited: {parent_node_token}")
        return []

    visited.add(parent_node_token)
    prefix = '|__' + '___' * depth + ' '
    docs = []

    # Fetch from root node
    if depth == 0 and parent_node_token:
        root_node = await fetch_doc_info(space_id, parent_node_token)
        doc = {
            'depth': depth,
            'title': root_node['title'],
            'node_token': root_node['node_token'],
            'parent_node_token': None,
            'obj_create_time': root_node['obj_create_time'],
            'obj_edit_time': root_node['obj_edit_time'],
            'obj_token': root_node['obj_token'],
            'children': [],
            'has_child': root_node['has_child'],
        }
        docs.append(doc)
    else:
        items = await feishu_fetch_with_iterator(
            'GET',
            f'/open-apis/wiki/v2/spaces/{space_id}/nodes',
            {
                'parent_node_token': parent_node_token,
                'page_size': 50,
            }
        )

        for item in items:
            if item['obj_type'] in ['doc', 'docx']:
                doc = {
                    'depth': depth,
                    'title': item['title'],
                    'node_token': item['node_token'],
                    'parent_node_token': parent_node_token,
                    'obj_create_time': item['obj_create_time'],
                    'obj_edit_time': item['obj_edit_time'],
                    'obj_token': item['obj_token'],
                    'children': [],
                    'has_child': item['has_child'],
                }
                docs.append(doc)

        print(f"{prefix}node: {parent_node_token or 'root'} {len(docs)} docs" if docs else '')

    # Ignore title `[hide]` or `[隐藏]`
    docs = [doc for doc in docs if '[hide]' not in doc['title'].lower() and '[隐藏]' not in doc['title'].lower()]

    for doc in docs:
        if doc['has_child']:
            doc['children'] = await fetch_all_docs(space_id, depth + 1, doc['node_token'], visited)

    return docs

async def main():
    space_id = '7345724206941732865'
    docs = await fetch_all_docs(space_id)
    print(docs)

# To run the async main function
asyncio.run(main())

