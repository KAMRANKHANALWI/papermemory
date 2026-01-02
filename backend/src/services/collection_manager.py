"""
Expanded collection manager with full CRUD operations
"""
import chromadb
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class CollectionManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    def get_collection(self, collection_name: str, embedding_function):
        """
        Get a specific collection's vectorstore by name.
        
        Args:
            collection_name: Name of the collection
            embedding_function: Embedding function to use (from chat_service)
            
        Returns:
            Chroma vectorstore instance or None if not found
        """
        try:
            from langchain_chroma import Chroma
            
            # Check if collection exists
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                print(f"Collection '{collection_name}' not found")
                return None
            
            # Load the collection
            vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=embedding_function,
            )
            
            return vectorstore
            
        except Exception as e:
            import traceback
            print(f"❌ Error getting collection {collection_name}: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def get_all_collections(self) -> List[Dict]:
        """Get all collections with metadata"""
        try:
            collections = self.chroma_client.list_collections()
            collection_info = []
            
            for col in collections:
                try:
                    collection = self.chroma_client.get_collection(col.name)
                    count = collection.count()
                    
                    # Get unique files
                    sample_result = collection.get(include=["metadatas"], limit=count)
                    unique_files = set()
                    if sample_result and sample_result.get("metadatas"):
                        for metadata in sample_result["metadatas"]:
                            if metadata and metadata.get("filename"):
                                unique_files.add(metadata["filename"])
                    
                    collection_info.append({
                        "name": col.name,
                        "chunk_count": count,
                        "file_count": len(unique_files)
                    })
                except:
                    collection_info.append({
                        "name": col.name,
                        "chunk_count": 0,
                        "file_count": 0
                    })
            
            return collection_info
        except:
            return []
        
    def get_all_collections_vectorstores(self, embedding_function) -> Dict:
        """
        Get all collections with their vectorstores.
        
        Args:
            embedding_function: Embedding function to use (from chat_service)
        
        Returns:
            Dict mapping collection_name -> vectorstore
        """
        try:
            from langchain_chroma import Chroma
            
            all_collections = {}
            collections = self.chroma_client.list_collections()
            
            for collection in collections:
                try:
                    vectorstore = Chroma(
                        client=self.chroma_client,
                        collection_name=collection.name,
                        embedding_function=embedding_function,
                    )
                    all_collections[collection.name] = vectorstore
                except Exception as e:
                    print(f"Error loading collection {collection.name}: {e}")
                    continue
            
            return all_collections
            
        except Exception as e:
            print(f"Error getting all collections: {e}")
            return {}
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.chroma_client.delete_collection(collection_name)
            return {"status": "success", "message": f"Collection {collection_name} deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def rename_collection(self, old_name: str, new_name: str) -> Dict:
        """
        Rename a collection by creating new and copying documents.
        
        Args:
            old_name: Current collection name
            new_name: New collection name
            
        Returns:
            Operation result dict
        """
        try:
            # Get old collection
            old_vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=old_name,
                embedding_function=self.embedding_model,
                persist_directory="data/chroma_db"
            )
            
            # Get all documents
            old_collection = self.chroma_client.get_collection(old_name)
            count = old_collection.count()
            
            if count == 0:
                return {"status": "error", "message": "Cannot rename empty collection"}
            
            # Get all data
            result = old_collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=count
            )
            
            # Create new collection with same documents
            new_collection = self.chroma_client.get_or_create_collection(new_name)
            
            # Add all documents to new collection
            if result["ids"]:
                new_collection.add(
                    ids=result["ids"],
                    documents=result["documents"],
                    metadatas=result["metadatas"],
                    embeddings=result["embeddings"]
                )
            
            # Delete old collection
            self.chroma_client.delete_collection(old_name)
            
            return {
                "status": "success",
                "message": f"Collection renamed from {old_name} to {new_name}",
                "new_name": new_name
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_pdf_from_collection(
        self, 
        collection_name: str, 
        filename: str
    ) -> Dict:
        """
        Delete all chunks of a specific PDF from collection.
        
        Args:
            collection_name: Collection name
            filename: PDF filename to delete
            
        Returns:
            Operation result dict
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Get all documents
            result = collection.get(include=["metadatas"])
            
            # Find IDs of chunks belonging to this file
            ids_to_delete = []
            for idx, metadata in enumerate(result["metadatas"]):
                if metadata and metadata.get("filename") == filename:
                    ids_to_delete.append(result["ids"][idx])
            
            if not ids_to_delete:
                return {
                    "status": "error",
                    "message": f"File {filename} not found in collection"
                }
            
            # Delete the chunks
            collection.delete(ids=ids_to_delete)
            
            return {
                "status": "success",
                "message": f"Deleted {len(ids_to_delete)} chunks of {filename}",
                "chunks_deleted": len(ids_to_delete)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def rename_pdf_in_collection(
        self,
        collection_name: str,
        old_filename: str,
        new_filename: str
    ) -> Dict:
        """
        Rename a PDF in collection by updating metadata.
        
        Args:
            collection_name: Collection name
            old_filename: Current filename
            new_filename: New filename
            
        Returns:
            Operation result dict
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Get all documents
            result = collection.get(include=["metadatas"])
            
            # Find and update metadata for this file
            updated_count = 0
            for idx, metadata in enumerate(result["metadatas"]):
                if metadata and metadata.get("filename") == old_filename:
                    # Update metadata
                    metadata["filename"] = new_filename
                    
                    # Update in collection
                    collection.update(
                        ids=[result["ids"][idx]],
                        metadatas=[metadata]
                    )
                    updated_count += 1
            
            if updated_count == 0:
                return {
                    "status": "error",
                    "message": f"File {old_filename} not found in collection"
                }
            
            return {
                "status": "success",
                "message": f"Renamed {old_filename} to {new_filename}",
                "chunks_updated": updated_count
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}