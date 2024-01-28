import argparse


class VectorStoreArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Create Vector Store from Web Scraped Documents.')
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--save_vector_store', type=bool, default=False, help='save the vectorstore to database')
        self.parser.add_argument('--search_url', type=str, default="https://genshin-impact.fandom.com/wiki/Furina", help='URL to search for documents')

    def parse_args(self):
        return self.parser.parse_args()