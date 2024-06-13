import os
import aiohttp
import asyncio

# Constants
FEISHU_OPENAPI_ENDPOINT = "https://open.feishu.cn/open-apis/wiki/v2/spaces"

async def get_wiki_node_list(space_id, headers, page_token=None, parent_node_token=None):
    url = f"{FEISHU_OPENAPI_ENDPOINT}/{space_id}/nodes?page_size=50"
    if page_token:
        url += f"&page_token={page_token}"
    if parent_node_token:
        url += f"&parent_node_token={parent_node_token}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                print(f"Failed to fetch from Feishu: {response.status}")
                return None
            result_data = await response.json()
    
    return result_data['data']

async def get_all_wiki_nodes(space_id, tenantAccessToken):
    try:
        nodes = []
        page_token = None
        has_more = True
        authorization = f"Bearer {tenantAccessToken}"
        headers = {
            'Authorization': authorization,
            'Content-Type': 'application/json; charset=utf-8',
            'User-Agent': 'feishu-pages',
        }

        while has_more and (page_token is None or page_token.strip()):
            # Fetch top-level nodes with pagination
            paged_result = await get_wiki_node_list(space_id, headers, page_token)
            if paged_result:
                nodes.extend(paged_result['items'])

                for item in paged_result['items']:
                    if item['has_child']:
                        child_nodes = await get_wiki_child_nodes(space_id, item['node_token'], headers)
                        nodes.extend(child_nodes)

                page_token = paged_result['page_token']
                has_more = paged_result['has_more']
            else:
                page_token = ""
                has_more = False

        return nodes

    except aiohttp.ClientError as ex:
        response_data = ''

        if isinstance(ex, aiohttp.ClientResponseError):
            response = ex.response
            response_data = await response.text()

        print(f"Request Exception!!!\nException Message: {ex.message},\nStack Trace: {ex.stack},\nResponse Data: {response_data}\n")
        raise

async def get_wiki_child_nodes(space_id, parent_node_token, headers):
    child_nodes = []
    page_token = None
    has_more = True

    while has_more and (page_token is None or page_token.strip()):
        paged_result = await get_wiki_node_list(space_id, headers, page_token, parent_node_token)
        child_nodes.extend(paged_result['items'])

        for item in paged_result['items']:
            if item['has_child']:
                grand_child_nodes = await get_wiki_child_nodes(space_id, item['node_token'], headers)
                child_nodes.extend(grand_child_nodes)

        page_token = paged_result['page_token']
        has_more = paged_result['has_more']

    return child_nodes

async def main():
    space_id = os.environ["SPACE_ID"]
    nodes = await get_all_wiki_nodes(space_id)
    print(nodes)

# asyncio.run(main())
