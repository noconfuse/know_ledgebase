# -*- coding: utf-8 -*-
"""
This module provides the functionality to vectorize a directory of documents.
"""

import os
import json
import hashlib
from typing import List, Dict, Optional

from llama_index.core import Document


from models.index_models import IndexInfo
from services.vector_store_builder import VectorStoreBuilder
from services.docling_json_enhancer import DoclingJsonEnhancer

class DirectoryVectorizer:
    """
    A class to handle the vectorization of a directory.
    """

    def __init__(self, vector_store_builder: VectorStoreBuilder):
        """
        Initializes the DirectoryVectorizer.

        Args:
            vector_store_builder (VectorStoreBuilder): An instance of VectorStoreBuilder.
        """
        self.vector_store_builder = vector_store_builder

    def vectorize_directory(self, directory_path: str, enhancement_config: Optional[Dict] = None) -> None:
        """
        Vectorizes the documents in a given directory.

        Args:
            directory_path (str): The path to the directory to be vectorized.
            enhancement_config (Optional[Dict], optional): Configuration for document enhancement. Defaults to None.
        """
        logger.info(f"Starting vectorization for directory: {directory_path}")

        documents = self._collect_documents(directory_path, enhancement_config)

        if not documents:
            logger.warning("No documents found to vectorize.")
            return

        self._process_documents(documents)

        logger.info(f"Finished vectorization for directory: {directory_path}")

    def _collect_documents(self, directory_path: str, enhancement_config: Optional[Dict]) -> List[Document]:
        """
        Collects documents from the directory, applying enhancement if configured.

        Args:
            directory_path (str): The path to the directory.
            enhancement_config (Optional[Dict]): Configuration for document enhancement.

        Returns:
            List[Document]: A list of collected documents.
        """
        documents = []
        enhancer = DoclingJsonEnhancer(enhancement_config) if enhancement_config else None

        for root, _, files in os.walk(directory_path):
            if enhancer:
                markdown_files = [f for f in files if f.endswith('.md')]
                for md_file in markdown_files:
                    base_name = os.path.splitext(md_file)[0]
                    json_file = f"{base_name}_docling.json"
                    if json_file in files:
                        json_path = os.path.join(root, json_file)
                        md_path = os.path.join(root, md_file)
                        enhanced_docs = enhancer.process_files(md_path, json_path)
                        documents.extend(enhanced_docs)
                        # Skip default processing for these files
                        files.remove(md_file)
                        files.remove(json_file)

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(Document(text=content, extra_info={"file_path": file_path}))
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Could not read file {file_path}: {e}")

        return documents

    def _process_documents(self, documents: List[Document]) -> None:
        """
        Processes and vectorizes a list of documents.

        Args:
            documents (List[Document]): The list of documents to process.
        """
        # This logic will be based on vector_store_builder
        # For now, we'll just log the document paths
        for doc in documents:
            file_path = doc.extra_info.get('file_path')
            if file_path:
                file_md5 = self._calculate_md5(file_path)
                existing_index = self._find_existing_index(file_md5)

                if existing_index:
                    logger.info(f"Updating index for document: {file_path}")
                    # Placeholder for update logic
                else:
                    logger.info(f"Creating new index for document: {file_path}")
                    # Placeholder for creation logic

    def _calculate_md5(self, file_path: str) -> str:
        """
        Calculates the MD5 hash of a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The MD5 hash of the file.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _find_existing_index(self, file_md5: str) -> Optional[IndexInfo]:
        """
        Finds an existing index by file MD5.

        Args:
            file_md5 (str): The MD5 hash of the file.

        Returns:
            Optional[IndexInfo]: The found index information, or None.
        """
        # This would typically involve a database lookup
        # Placeholder implementation
        return None