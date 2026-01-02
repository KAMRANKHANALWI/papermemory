"""
Helper utilities for PDF Selection feature
"""
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class SelectionAnalyzer:
    """Analyze and provide insights about PDF selections"""
    
    @staticmethod
    def analyze_selection_overlap(
        session_id_1: str,
        session_id_2: str,
        pdf_selection_service
    ) -> Dict:
        """
        Analyze overlap between two selection sessions.
        
        Returns:
            Dict with overlap statistics
        """
        session1 = pdf_selection_service.get_selected_pdfs(session_id_1)
        session2 = pdf_selection_service.get_selected_pdfs(session_id_2)
        
        if not session1 or not session2:
            return {"error": "One or both sessions not found"}
        
        pdfs1 = {(pdf["filename"], pdf["collection_name"]) 
                 for pdf in session1["selected_pdfs"]}
        pdfs2 = {(pdf["filename"], pdf["collection_name"]) 
                 for pdf in session2["selected_pdfs"]}
        
        overlap = pdfs1.intersection(pdfs2)
        unique_to_1 = pdfs1 - pdfs2
        unique_to_2 = pdfs2 - pdfs1
        
        return {
            "total_session1": len(pdfs1),
            "total_session2": len(pdfs2),
            "overlap_count": len(overlap),
            "overlap_percentage": (len(overlap) / max(len(pdfs1), 1)) * 100,
            "unique_to_session1": len(unique_to_1),
            "unique_to_session2": len(unique_to_2),
            "overlapping_pdfs": [
                {"filename": f, "collection": c} 
                for f, c in overlap
            ]
        }
    
    @staticmethod
    def get_collection_distribution(session_id: str, pdf_selection_service) -> Dict:
        """
        Get distribution of selected PDFs across collections.
        
        Returns:
            Dict with collection distribution stats
        """
        session = pdf_selection_service.get_selected_pdfs(session_id)
        
        if not session:
            return {"error": "Session not found"}
        
        distribution = defaultdict(int)
        for pdf in session["selected_pdfs"]:
            distribution[pdf["collection_name"]] += 1
        
        total = session["total_selected"]
        
        return {
            "total_selected": total,
            "num_collections": len(distribution),
            "distribution": dict(distribution),
            "percentages": {
                coll: (count / total) * 100 
                for coll, count in distribution.items()
            }
        }
    
    @staticmethod
    def suggest_related_pdfs(
        session_id: str,
        pdf_selection_service,
        all_collections: Dict,
        max_suggestions: int = 5
    ) -> List[Dict]:
        """
        Suggest related PDFs based on current selection.
        
        This is a simple implementation - can be enhanced with
        semantic similarity, topic modeling, etc.
        """
        session = pdf_selection_service.get_selected_pdfs(session_id)
        
        if not session or not session["selected_pdfs"]:
            return []
        
        # Get collections that have selected PDFs
        active_collections = session["collections_involved"]
        
        # Get all PDFs from these collections
        suggestions = []
        
        for coll_name in active_collections:
            if coll_name not in all_collections:
                continue
            
            vectorstore = all_collections[coll_name]
            collection = vectorstore._collection
            
            # Get all PDFs in this collection
            result = collection.get(include=["metadatas"], limit=10000)
            
            seen_pdfs = set()
            selected_pdfs = {
                pdf["filename"] 
                for pdf in session["selected_pdfs"]
                if pdf["collection_name"] == coll_name
            }
            
            for metadata in result.get("metadatas", []):
                filename = metadata.get("filename")
                
                if not filename or filename in seen_pdfs or filename in selected_pdfs:
                    continue
                
                seen_pdfs.add(filename)
                
                suggestions.append({
                    "filename": filename,
                    "collection_name": coll_name,
                    "title": metadata.get("title", "No Title"),
                    "reason": f"From same collection as selected PDFs"
                })
                
                if len(suggestions) >= max_suggestions:
                    break
            
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions[:max_suggestions]


class SelectionValidator:
    """Validate PDF selections and provide warnings"""
    
    @staticmethod
    def validate_selection_size(
        session_id: str,
        pdf_selection_service,
        max_recommended: int = 10,
        max_chunks: int = 1000
    ) -> Dict:
        """
        Validate if selection is within recommended limits.
        
        Returns:
            Dict with validation results and warnings
        """
        session = pdf_selection_service.get_selected_pdfs(session_id)
        
        if not session:
            return {"valid": False, "error": "Session not found"}
        
        total_selected = session["total_selected"]
        total_chunks = sum(pdf.get("chunks", 0) for pdf in session["selected_pdfs"])
        
        warnings = []
        
        if total_selected > max_recommended:
            warnings.append(
                f"You have selected {total_selected} PDFs. "
                f"Consider limiting to {max_recommended} for better performance."
            )
        
        if total_chunks > max_chunks:
            warnings.append(
                f"Total chunks ({total_chunks}) exceeds recommended limit ({max_chunks}). "
                "Search may be slow or results may be less relevant."
            )
        
        return {
            "valid": True,
            "total_selected": total_selected,
            "total_chunks": total_chunks,
            "within_limits": len(warnings) == 0,
            "warnings": warnings
        }
    
    @staticmethod
    def check_collection_availability(
        session_id: str,
        pdf_selection_service,
        all_collections: Dict
    ) -> Dict:
        """
        Check if all selected collections are still available.
        
        Returns:
            Dict with availability status
        """
        session = pdf_selection_service.get_selected_pdfs(session_id)
        
        if not session:
            return {"error": "Session not found"}
        
        unavailable = []
        
        for coll_name in session["collections_involved"]:
            if coll_name not in all_collections:
                unavailable.append(coll_name)
        
        return {
            "all_available": len(unavailable) == 0,
            "unavailable_collections": unavailable,
            "message": (
                "All collections are available" 
                if not unavailable 
                else f"Warning: {len(unavailable)} collection(s) not available"
            )
        }


class SelectionExporter:
    """Export and import selection configurations"""
    
    @staticmethod
    def export_to_preset(
        session_id: str,
        pdf_selection_service,
        preset_name: str
    ) -> Dict:
        """
        Export selection as a named preset.
        
        Returns:
            Dict with preset configuration
        """
        session = pdf_selection_service.get_selected_pdfs(session_id)
        
        if not session:
            return {"error": "Session not found"}
        
        preset = {
            "preset_name": preset_name,
            "created_at": session["created_at"],
            "pdfs": [
                {
                    "filename": pdf["filename"],
                    "collection_name": pdf["collection_name"]
                }
                for pdf in session["selected_pdfs"]
            ],
            "metadata": {
                "total_pdfs": session["total_selected"],
                "collections": session["collections_involved"]
            }
        }
        
        return preset
    
    @staticmethod
    def import_from_preset(
        session_id: str,
        pdf_selection_service,
        preset: Dict,
        all_collections: Dict
    ) -> Tuple[bool, str, List[str]]:
        """
        Import selection from a preset.
        
        Returns:
            Tuple of (success, message, failed_pdfs)
        """
        if "pdfs" not in preset:
            return False, "Invalid preset format", []
        
        failed = []
        success_count = 0
        
        for pdf_info in preset["pdfs"]:
            filename = pdf_info.get("filename")
            collection_name = pdf_info.get("collection_name")
            
            if not filename or not collection_name:
                continue
            
            if collection_name not in all_collections:
                failed.append(f"{collection_name}/{filename} (collection not found)")
                continue
            
            vectorstore = all_collections[collection_name]
            success, _ = pdf_selection_service.select_pdf(
                session_id=session_id,
                filename=filename,
                collection_name=collection_name,
                vectorstore=vectorstore
            )
            
            if success:
                success_count += 1
            else:
                failed.append(f"{collection_name}/{filename}")
        
        total = len(preset["pdfs"])
        message = f"Imported {success_count}/{total} PDFs"
        
        return success_count > 0, message, failed


def format_selection_summary(session_data: Dict) -> str:
    """
    Format selection data into a human-readable summary.
    
    Args:
        session_data: Session data from get_selected_pdfs()
        
    Returns:
        Formatted summary string
    """
    if not session_data:
        return "No PDFs selected"
    
    lines = [
        f"📚 Selection Summary",
        f"─" * 50,
        f"Total PDFs: {session_data['total_selected']}",
        f"Collections: {', '.join(session_data['collections_involved'])}",
        f"",
        f"Selected Documents:",
    ]
    
    # Group by collection
    by_collection = defaultdict(list)
    for pdf in session_data["selected_pdfs"]:
        by_collection[pdf["collection_name"]].append(pdf)
    
    for coll_name, pdfs in by_collection.items():
        lines.append(f"\n  📁 {coll_name} ({len(pdfs)} PDFs)")
        for pdf in pdfs:
            lines.append(
                f"    • {pdf['filename']}"
                f" ({pdf.get('pages', 0)} pages, {pdf.get('chunks', 0)} chunks)"
            )
    
    return "\n".join(lines)
