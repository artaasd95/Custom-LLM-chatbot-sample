#!/usr/bin/env python3
"""Tests for the data processor module."""

import os
import sys
import tempfile
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import ConfigManager
from src.data.data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager for testing."""
        config = Mock()
        config.config = Mock()
        config.config.data = Mock()
        config.config.data.max_seq_length = 512
        config.config.data.chunk_size = 1000
        config.config.data.overlap_size = 100
        config.config.data.train_split = 0.8
        config.config.data.eval_split = 0.1
        config.config.data.test_split = 0.1
        config.config.data.cleaning_enabled = True
        config.config.data.min_text_length = 10
        config.config.data.max_text_length = 2048
        return config
    
    @pytest.fixture
    def data_processor(self, config_manager):
        """Create a DataProcessor instance for testing."""
        return DataProcessor(config_manager)
    
    def test_init(self, config_manager):
        """Test DataProcessor initialization."""
        processor = DataProcessor(config_manager)
        assert processor.config == config_manager
        assert processor.max_seq_length == 512
        assert processor.chunk_size == 1000
    
    def test_detect_data_format_text(self, data_processor):
        """Test detection of text format."""
        data = {"text": "This is a simple text."}
        format_type = data_processor.detect_data_format(data)
        assert format_type == "text"
    
    def test_detect_data_format_instruction(self, data_processor):
        """Test detection of instruction format."""
        data = {
            "instruction": "Translate to French:",
            "input": "Hello",
            "output": "Bonjour"
        }
        format_type = data_processor.detect_data_format(data)
        assert format_type == "instruction"
    
    def test_detect_data_format_conversation(self, data_processor):
        """Test detection of conversation format."""
        data = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        format_type = data_processor.detect_data_format(data)
        assert format_type == "conversation"
    
    def test_detect_data_format_dpo(self, data_processor):
        """Test detection of DPO format."""
        data = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "AI is a robot."
        }
        format_type = data_processor.detect_data_format(data)
        assert format_type == "dpo"
    
    def test_detect_data_format_input_output(self, data_processor):
        """Test detection of input-output format."""
        data = {
            "input": "What is 2+2?",
            "output": "4"
        }
        format_type = data_processor.detect_data_format(data)
        assert format_type == "input_output"
    
    def test_clean_text(self, data_processor):
        """Test text cleaning functionality."""
        dirty_text = "  This is\ta\ntest\r\nwith   extra   spaces.  "
        clean_text = data_processor.clean_text(dirty_text)
        expected = "This is a test with extra spaces."
        assert clean_text == expected
    
    def test_clean_text_disabled(self, data_processor):
        """Test text cleaning when disabled."""
        data_processor.cleaning_enabled = False
        dirty_text = "  This is\ta\ntest\r\nwith   extra   spaces.  "
        clean_text = data_processor.clean_text(dirty_text)
        assert clean_text == dirty_text
    
    def test_chunk_text(self, data_processor):
        """Test text chunking functionality."""
        long_text = "word " * 2000  # Create a long text
        chunks = data_processor.chunk_text(long_text)
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= data_processor.chunk_size for chunk in chunks)
    
    def test_process_text_format(self, data_processor):
        """Test processing of text format data."""
        data = {"text": "This is a test text."}
        result = data_processor.process_text_format(data)
        assert result == "This is a test text."
    
    def test_process_instruction_format(self, data_processor):
        """Test processing of instruction format data."""
        data = {
            "instruction": "Translate to French:",
            "input": "Hello",
            "output": "Bonjour"
        }
        result = data_processor.process_instruction_format(data)
        expected = "Instruction: Translate to French:\nInput: Hello\nOutput: Bonjour"
        assert result == expected
    
    def test_process_conversation_format(self, data_processor):
        """Test processing of conversation format data."""
        data = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        result = data_processor.process_conversation_format(data)
        expected = "User: Hello\nAssistant: Hi there!"
        assert result == expected
    
    def test_process_dpo_format(self, data_processor):
        """Test processing of DPO format data."""
        data = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "AI is a robot."
        }
        result = data_processor.process_dpo_format(data)
        expected = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "AI is a robot."
        }
        assert result == expected
    
    def test_process_input_output_format(self, data_processor):
        """Test processing of input-output format data."""
        data = {
            "input": "What is 2+2?",
            "output": "4"
        }
        result = data_processor.process_input_output_format(data)
        expected = "Input: What is 2+2?\nOutput: 4"
        assert result == expected
    
    def test_process_single_item(self, data_processor):
        """Test processing of a single data item."""
        data = {"text": "This is a test."}
        result = data_processor.process_single_item(data, "text")
        assert result == "This is a test."
    
    def test_split_data(self, data_processor):
        """Test data splitting functionality."""
        data = [f"Item {i}" for i in range(100)]
        train, eval_data, test = data_processor.split_data(data)
        
        assert len(train) == 80  # 80% of 100
        assert len(eval_data) == 10  # 10% of 100
        assert len(test) == 10  # 10% of 100
        
        # Check no overlap
        all_items = set(train + eval_data + test)
        assert len(all_items) == 100
    
    def test_load_json_file(self, data_processor):
        """Test loading JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [{"text": "Test 1"}, {"text": "Test 2"}]
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = data_processor.load_json_file(temp_path)
            assert result == test_data
        finally:
            os.unlink(temp_path)
    
    def test_load_jsonl_file(self, data_processor):
        """Test loading JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "Test 1"}\n')
            f.write('{"text": "Test 2"}\n')
            temp_path = f.name
        
        try:
            result = data_processor.load_jsonl_file(temp_path)
            expected = [{"text": "Test 1"}, {"text": "Test 2"}]
            assert result == expected
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_file(self, data_processor):
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('text\n')
            f.write('Test 1\n')
            f.write('Test 2\n')
            temp_path = f.name
        
        try:
            result = data_processor.load_csv_file(temp_path)
            expected = [{"text": "Test 1"}, {"text": "Test 2"}]
            assert result == expected
        finally:
            os.unlink(temp_path)
    
    def test_load_txt_file(self, data_processor):
        """Test loading TXT file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('This is line 1\n')
            f.write('This is line 2\n')
            temp_path = f.name
        
        try:
            result = data_processor.load_txt_file(temp_path)
            expected = [{"text": "This is line 1"}, {"text": "This is line 2"}]
            assert result == expected
        finally:
            os.unlink(temp_path)
    
    @patch('src.data.data_processor.PyPDF2')
    def test_load_pdf_file(self, mock_pypdf2, data_processor):
        """Test loading PDF file."""
        # Mock PDF reader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        
        try:
            result = data_processor.load_pdf_file(temp_path)
            expected = [{"text": "PDF content"}]
            assert result == expected
        finally:
            os.unlink(temp_path)
    
    @patch('src.data.data_processor.Document')
    def test_load_docx_file(self, mock_document, data_processor):
        """Test loading DOCX file."""
        # Mock document
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "DOCX content"
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            temp_path = f.name
        
        try:
            result = data_processor.load_docx_file(temp_path)
            expected = [{"text": "DOCX content"}]
            assert result == expected
        finally:
            os.unlink(temp_path)
    
    def test_load_file_unsupported_format(self, data_processor):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                data_processor.load_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_process_data_list(self, data_processor):
        """Test processing a list of data items."""
        data = [
            {"text": "Item 1"},
            {"text": "Item 2"},
            {"instruction": "Test", "input": "Input", "output": "Output"}
        ]
        
        result = data_processor.process_data(data)
        
        assert len(result) == 3
        assert result[0] == "Item 1"
        assert result[1] == "Item 2"
        assert "Instruction: Test" in result[2]
    
    def test_process_data_with_splitting(self, data_processor):
        """Test processing data with train/eval/test splitting."""
        data = [{"text": f"Item {i}"} for i in range(100)]
        
        result = data_processor.process_data(data, split_data=True)
        
        assert "train" in result
        assert "eval" in result
        assert "test" in result
        assert len(result["train"]) == 80
        assert len(result["eval"]) == 10
        assert len(result["test"]) == 10
    
    def test_process_directory(self, data_processor):
        """Test processing a directory of files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, 'w') as f:
                json.dump([{"text": "JSON content"}], f)
            
            txt_file = os.path.join(temp_dir, "test.txt")
            with open(txt_file, 'w') as f:
                f.write("TXT content")
            
            result = data_processor.process_directory(temp_dir)
            
            assert len(result) >= 2  # At least 2 items from our files
            assert "JSON content" in result
            assert "TXT content" in result
    
    def test_get_stats(self, data_processor):
        """Test getting processing statistics."""
        # Process some data to generate stats
        data = [{"text": f"Item {i}"} for i in range(10)]
        data_processor.process_data(data)
        
        stats = data_processor.get_stats()
        
        assert "total_items" in stats
        assert "processed_items" in stats
        assert "failed_items" in stats
        assert "formats_detected" in stats
        assert stats["total_items"] == 10
        assert stats["processed_items"] == 10
    
    def test_error_handling_invalid_json(self, data_processor):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{invalid json}')
            temp_path = f.name
        
        try:
            result = data_processor.load_json_file(temp_path)
            assert result == []  # Should return empty list on error
        finally:
            os.unlink(temp_path)
    
    def test_text_length_filtering(self, data_processor):
        """Test filtering of texts based on length."""
        data = [
            {"text": "Short"},  # Too short
            {"text": "This is a good length text for testing."},  # Good length
            {"text": "x" * 3000}  # Too long
        ]
        
        result = data_processor.process_data(data)
        
        # Should only keep the middle item
        assert len(result) == 1
        assert "good length" in result[0]
    
    def test_normalize_text(self, data_processor):
        """Test text normalization."""
        text = "This\u2019s a test with unicode quotes\u201d"
        normalized = data_processor.normalize_text(text)
        assert "'" in normalized
        assert '"' in normalized
        assert "\u2019" not in normalized
    
    def test_process_with_chunking(self, data_processor):
        """Test processing with text chunking enabled."""
        # Create a very long text that will need chunking
        long_text = "word " * 2000
        data = [{"text": long_text}]
        
        result = data_processor.process_data(data)
        
        # Should have multiple chunks
        assert len(result) > 1
        assert all(len(chunk.split()) <= data_processor.chunk_size for chunk in result)


if __name__ == "__main__":
    pytest.main([__file__])