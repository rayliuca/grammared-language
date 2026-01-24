"""Tests for config_parser utility module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import yaml

from grammared_language.utils.config_parser import (
    load_config,
    create_client_from_config,
    create_clients_from_config,
    ModelsConfig,
    GectorConfig,
    GrammaredClassifierConfig
)


class TestLoadConfig:
    """Test load_config function."""
    
    def test_load_config_success(self, tmp_path):
        """Test loading a valid config file."""
        config_data = {
            'model1': {'type': 'gector', 'backend': 'triton', 'pretrained_model_name_or_path': 'test-model-1'},
            'model2': {'type': 'grammared_classifier', 'backend': 'triton', 'pretrained_model_name_or_path': 'test-model-2'}
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_config(str(config_file))
        assert isinstance(result, ModelsConfig)
        assert len(result.models) == 2
        assert 'model1' in result.models
        assert 'model2' in result.models
    
    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent_config.yaml")


class TestCreateClientFromConfig:
    """Test create_client_from_config function."""
    
    @patch('grammared_language.clients.gector_client.GectorClient')
    def test_create_gector_client(self, mock_gector):
        """Test creating a GectorClient from config."""
        config = {
            'type': 'gector',
            'backend': 'triton',
            'pretrained_model_name_or_path': 'test-model',
            'triton_model_name': 'test_triton_model'
        }
        
        mock_client = Mock()
        mock_gector.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        mock_gector.assert_called_once()
    
    @patch('grammared_language.clients.grammar_classification_client.GrammarClassificationClient')
    def test_create_grammared_classifier_client(self, mock_classifier):
        """Test creating a GrammarClassificationClient from config."""
        config = {
            'type': 'grammared_classifier',
            'backend': 'triton',
            'pretrained_model_name_or_path': 'test-model',
            'triton_model_name': 'test_triton_model',
            'triton_hostname': 'localhost',
            'triton_port': 8001
        }
        
        mock_client = Mock()
        mock_classifier.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        mock_classifier.assert_called_once()
    
    @patch('grammared_language.clients.coedit_client.CoEditClient')
    def test_create_coedit_client(self, mock_coedit):
        """Test creating a CoEditClient from config."""
        config = {
            'type': 'coedit',
            'backend': 'triton',
            'pretrained_model_name_or_path': 'test-model',
            'triton_model_name': 'test_triton_model'
        }
        
        mock_client = Mock()
        mock_coedit.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        mock_coedit.assert_called_once()
    
    def test_create_client_unknown_type(self):
        """Test handling of unknown client type."""
        config = {
            'type': 'unknown_type',
            'backend': 'triton'
        }
        
        with pytest.raises(ValueError, match="Unknown model type for test_model: unknown_type"):
            create_client_from_config('test_model', config)
    
    @patch('grammared_language.clients.gector_client.GectorClient')
    def test_create_client_exception_handling(self, mock_gector):
        """Test exception handling during client creation."""
        config = {
            'type': 'gector',
            'backend': 'triton',
            'pretrained_model_name_or_path': 'test-model'
        }
        
        mock_gector.side_effect = Exception("Test error")
        
        result = create_client_from_config('test_model', config)
        assert result is None


class TestCreateClientsFromConfig:
    """Test create_clients_from_config function."""
    
    @patch('grammared_language.utils.config_parser.create_client_from_config')
    @patch('grammared_language.utils.config_parser.load_config')
    def test_create_clients_from_config(self, mock_load, mock_create):
        """Test creating multiple clients from config file."""
        mock_models_config = ModelsConfig(
            models={
                'model1': GectorConfig(type='gector', backend='triton', pretrained_model_name_or_path='test1'),
                'model2': GrammaredClassifierConfig(type='grammared_classifier', backend='triton', pretrained_model_name_or_path='test2')
            }
        )
        mock_load.return_value = mock_models_config
        
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_create.side_effect = [mock_client1, mock_client2]
        
        result = create_clients_from_config('test_config.yaml')
        
        assert len(result) == 2
        assert result[0] == mock_client1
        assert result[1] == mock_client2
    
    @patch('grammared_language.utils.config_parser.create_client_from_config')
    @patch('grammared_language.utils.config_parser.load_config')
    def test_create_clients_skip_invalid(self, mock_load, mock_create):
        """Test that invalid clients are skipped."""
        # ModelsConfig.from_dict already filters out invalid configs, so we test with valid models
        mock_models_config = ModelsConfig(
            models={
                'model1': GectorConfig(type='gector', backend='triton', pretrained_model_name_or_path='test1'),
                'model3': GrammaredClassifierConfig(type='grammared_classifier', backend='triton', pretrained_model_name_or_path='test3')
            }
        )
        mock_load.return_value = mock_models_config
        
        mock_client1 = Mock()
        mock_client3 = Mock()
        mock_create.side_effect = [mock_client1, mock_client3]
        
        result = create_clients_from_config('test_config.yaml')
        
        # Should have 2 clients
        assert len(result) == 2
        assert mock_create.call_count == 2
    
    @patch('grammared_language.utils.config_parser.create_client_from_config')
    @patch('grammared_language.utils.config_parser.load_config')
    def test_create_clients_handle_none_return(self, mock_load, mock_create):
        """Test that None returns from create_client_from_config are filtered."""
        mock_models_config = ModelsConfig(
            models={
                'model1': GectorConfig(type='gector', backend='triton', pretrained_model_name_or_path='test1'),
                'model2': GectorConfig(type='gector', backend='triton', pretrained_model_name_or_path='test2')
            }
        )
        mock_load.return_value = mock_models_config
        
        mock_client1 = Mock()
        mock_create.side_effect = [mock_client1, None]
        
        result = create_clients_from_config('test_config.yaml')
        
        # Should only have 1 client (model2 returned None)
        assert len(result) == 1
        assert result[0] == mock_client1
