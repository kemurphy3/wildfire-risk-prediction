"""
Tests for NEON data collection module.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.data_collection import NEONDataCollector


class TestNEONDataCollector:
    """Test cases for NEON data collection."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = NEONDataCollector()
        assert collector.api_token is None or isinstance(collector.api_token, str)
        assert collector.cache_dir.exists()
    
    def test_fire_prone_sites(self):
        """Test that fire-prone sites are properly defined."""
        collector = NEONDataCollector()
        assert len(collector.FIRE_PRONE_SITES) > 0
        assert "SJER" in collector.FIRE_PRONE_SITES
    
    def test_fire_relevant_products(self):
        """Test that fire-relevant products are properly defined."""
        collector = NEONDataCollector()
        assert len(collector.FIRE_RELEVANT_PRODUCTS) > 0
        assert "DP1.00041.001" in collector.FIRE_RELEVANT_PRODUCTS
    
    @patch('requests.Session.get')
    def test_get_sites_info(self, mock_get):
        """Test site information retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"siteCode": "SJER", "siteName": "San Joaquin", "stateCode": "CA"},
                {"siteCode": "SOAP", "siteName": "Soaproot", "stateCode": "CA"}
            ]
        }
        mock_get.return_value = mock_response
        
        collector = NEONDataCollector()
        sites_df = collector.get_sites_info(site_codes=["SJER", "SOAP"])
        
        assert len(sites_df) == 2
        assert "SJER" in sites_df["siteCode"].values
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created."""
        collector = NEONDataCollector(cache_dir="tests/temp_cache")
        assert collector.cache_dir.exists()
        
        # Cleanup
        import shutil
        shutil.rmtree("tests/temp_cache")


if __name__ == "__main__":
    pytest.main([__file__])