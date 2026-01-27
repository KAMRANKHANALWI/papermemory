"""
RAG Evaluation Dataset Processor
Converts gut microbiome Q&A JSON files into clean, uniform dataset for Ragas evaluation
Author: Kamran
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re

class RAGDatasetProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize the processor
        
        Args:
            data_dir: Path to directory containing JSON files
        """
        self.data_dir = Path(data_dir)
        self.json_files = sorted(self.data_dir.glob("*.json"))
        print(f"Found {len(self.json_files)} JSON files")
        
    def load_json_file(self, filepath: Path) -> Dict:
        """Load a single JSON file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
            return None
    
    def extract_test_cases(self, data: Dict, source_file: str) -> List[Dict]:
        """Extract test cases from JSON data"""
        test_cases = []
        
        if 'test_cases' not in data or not data['test_cases']:
            return test_cases
        
        for tc in data['test_cases']:
            test_case = {
                'id': tc.get('id', ''),
                'question': tc.get('query', ''),
                'reference': tc.get('ground_truth_answer', ''),
                'source_document': data.get('source_document', source_file),
                'expected_documents': tc.get('expected_documents', []),
                'difficulty': tc.get('difficulty', 'unknown'),
                'category': tc.get('category', 'unknown'),
                'question_type': 'open_ended'
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def extract_mcq_questions(self, data: Dict, source_file: str) -> List[Dict]:
        """Extract MCQ questions from JSON data"""
        mcqs = []
        
        if 'mcq_questions' not in data or not data['mcq_questions']:
            return mcqs
        
        for mcq in data['mcq_questions']:
            # Convert MCQ to open-ended format for Ragas
            question = mcq.get('question', '')
            options = mcq.get('options', [])
            correct_answer = mcq.get('correct_answer', '')
            explanation = mcq.get('explanation', '')
            
            # Format options nicely
            options_text = "\n".join(options)
            
            # Create reference answer with correct option and explanation
            reference = f"{correct_answer}. {explanation}"
            
            mcq_case = {
                'id': mcq.get('id', ''),
                'question': f"{question}\n\nOptions:\n{options_text}",
                'reference': reference,
                'source_document': data.get('source_document', source_file),
                'expected_documents': mcq.get('expected_documents', []),
                'difficulty': mcq.get('difficulty', 'unknown'),
                'category': mcq.get('category', 'unknown'),
                'question_type': 'mcq',
                'correct_option': correct_answer,
                'explanation': explanation
            }
            mcqs.append(mcq_case)
        
        return mcqs
    
    def process_all_files(self, include_mcqs: bool = True) -> pd.DataFrame:
        """
        Process all JSON files and create unified dataset
        
        Args:
            include_mcqs: Whether to include MCQ questions in dataset
        
        Returns:
            pandas DataFrame with processed data
        """
        all_data = []
        
        for json_file in self.json_files:
            print(f"Processing: {json_file.name}")
            
            data = self.load_json_file(json_file)
            if data is None:
                continue
            
            # Extract test cases
            test_cases = self.extract_test_cases(data, json_file.name)
            all_data.extend(test_cases)
            print(f"  - Extracted {len(test_cases)} test cases")
            
            # Extract MCQ questions if requested
            if include_mcqs:
                mcqs = self.extract_mcq_questions(data, json_file.name)
                all_data.extend(mcqs)
                print(f"  - Extracted {len(mcqs)} MCQ questions")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"\n{'='*60}")
        print(f"Total records processed: {len(df)}")
        print(f"{'='*60}")
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset"""
        print("\nValidating dataset...")
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Remove rows with missing critical fields
        critical_fields = ['id', 'question', 'reference']
        before_count = len(df)
        df = df.dropna(subset=critical_fields)
        after_count = len(df)
        
        if before_count != after_count:
            print(f"\nRemoved {before_count - after_count} rows with missing critical fields")
        
        # Remove duplicates based on ID
        # before_count = len(df)
        # df = df.drop_duplicates(subset=['id'], keep='first')
        # after_count = len(df)
        
        # if before_count != after_count:
        #     print(f"Removed {before_count - after_count} duplicate IDs")
        
        # Clean text fields
        text_fields = ['question', 'reference']
        for field in text_fields:
            df[field] = df[field].str.strip()
            df[field] = df[field].str.replace(r'\s+', ' ', regex=True)
        
        # Validate difficulty levels
        valid_difficulties = ['easy', 'medium', 'hard', 'unknown']
        invalid_difficulties = df[~df['difficulty'].isin(valid_difficulties)]
        if len(invalid_difficulties) > 0:
            print(f"\nWarning: Found {len(invalid_difficulties)} rows with invalid difficulty levels")
            df.loc[~df['difficulty'].isin(valid_difficulties), 'difficulty'] = 'unknown'
        
        print("\nValidation complete!")
        return df
    
    def create_ragas_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset in Ragas format
        
        Ragas expects:
        - question: The question/query
        - reference: Ground truth answer
        - contexts: List of context strings (optional, for context_precision/recall)
        - answer: The generated answer (to be filled during evaluation)
        """
        ragas_df = df[['id', 'question', 'reference']].copy()
        
        # Add empty contexts and answer columns for Ragas evaluation
        ragas_df['contexts'] = [[] for _ in range(len(ragas_df))]
        ragas_df['answer'] = ''
        
        # Add metadata columns
        ragas_df['difficulty'] = df['difficulty']
        ragas_df['category'] = df['category']
        ragas_df['question_type'] = df['question_type']
        ragas_df['source_document'] = df['source_document']
        
        return ragas_df
    
    def get_dataset_statistics(self, df: pd.DataFrame):
        """Print comprehensive dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal Questions: {len(df)}")
        
        print("\nBy Question Type:")
        print(df['question_type'].value_counts())
        
        print("\nBy Difficulty:")
        print(df['difficulty'].value_counts())
        
        print("\nBy Category:")
        print(df['category'].value_counts())
        
        print("\nBy Source Document:")
        print(df['source_document'].value_counts().head(10))
        
        print("\nQuestion Length Statistics:")
        df['question_length'] = df['question'].str.len()
        print(df['question_length'].describe())
        
        print("\nReference Answer Length Statistics:")
        df['reference_length'] = df['reference'].str.len()
        print(df['reference_length'].describe())
        
        print("\n" + "="*60)
    
    def save_datasets(self, df: pd.DataFrame, ragas_df: pd.DataFrame, output_dir: str = "./output"):
        """Save processed datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save full dataset with all metadata
        full_path = output_path / "gut_microbiome_full_dataset.csv"
        df.to_csv(full_path, index=False)
        print(f"\nSaved full dataset: {full_path}")
        
        # Save Ragas-ready dataset
        ragas_path = output_path / "gut_microbiome_ragas_format.csv"
        ragas_df.to_csv(ragas_path, index=False)
        print(f"Saved Ragas format: {ragas_path}")
        
        # Save as JSON for HuggingFace
        hf_path = output_path / "gut_microbiome_dataset.json"
        df.to_json(hf_path, orient='records', indent=2)
        print(f"Saved HuggingFace format: {hf_path}")
        
        # Save by difficulty level
        for difficulty in df['difficulty'].unique():
            diff_df = ragas_df[ragas_df['difficulty'] == difficulty]
            diff_path = output_path / f"gut_microbiome_{difficulty}.csv"
            diff_df.to_csv(diff_path, index=False)
            print(f"Saved {difficulty} questions: {diff_path} ({len(diff_df)} questions)")
        
        # Save by question type
        for qtype in df['question_type'].unique():
            type_df = ragas_df[ragas_df['question_type'] == qtype]
            type_path = output_path / f"gut_microbiome_{qtype}.csv"
            type_df.to_csv(type_path, index=False)
            print(f"Saved {qtype} questions: {type_path} ({len(type_df)} questions)")
        
        return full_path, ragas_path, hf_path


def main():
    """Main execution function"""
    
    # Configuration
    DATA_DIR = "./Eval_Data"
    OUTPUT_DIR = "./Processed_Eval_Data"
    INCLUDE_MCQS = True  # Set to False if you only want open-ended questions
    
    print("="*60)
    print("RAG EVALUATION DATASET PROCESSOR")
    print("Gut Microbiome Research Dataset")
    print("="*60)
    
    # Initialize processor
    processor = RAGDatasetProcessor(DATA_DIR)
    
    # Process all files
    print("\n1. Processing JSON files...")
    df = processor.process_all_files(include_mcqs=INCLUDE_MCQS)
    
    # Validate dataset
    print("\n2. Validating dataset...")
    df = processor.validate_dataset(df)
    
    # Create Ragas format
    print("\n3. Creating Ragas-compatible format...")
    ragas_df = processor.create_ragas_format(df)
    
    # Get statistics
    print("\n4. Generating statistics...")
    processor.get_dataset_statistics(df)
    
    # Save datasets
    print("\n5. Saving datasets...")
    full_path, ragas_path, hf_path = processor.save_datasets(df, ragas_df, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nDatasets saved in: {OUTPUT_DIR}")
    print(f"- Full dataset: {full_path.name}")
    print(f"- Ragas format: {ragas_path.name}")
    print(f"- HuggingFace JSON: {hf_path.name}")
    
    return df, ragas_df


if __name__ == "__main__":
    df, ragas_df = main()