"""
Interfaces for data handling following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict, Any, List

class DatasetInterface(ABC):
    """Interface for dataset handling."""
    
    @abstractmethod
    def load_dataset(self, data_dir: str) -> pd.DataFrame:
        """
        Load a dataset from a directory.
        
        Args:
            data_dir: Path to the dataset directory
            
        Returns:
            DataFrame with dataset information
        """
        pass
    
    @abstractmethod
    def create_tf_dataset(self, dataset_df: pd.DataFrame, batch_size: int = 32, 
                         training: bool = True) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a DataFrame.
        
        Args:
            dataset_df: DataFrame with dataset information
            batch_size: Batch size for the dataset
            training: Whether the dataset is for training
            
        Returns:
            TensorFlow dataset
        """
        pass
    
    @abstractmethod
    def split_train_val(self, dataset_df: pd.DataFrame, 
                       val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataset into training and validation sets.
        
        Args:
            dataset_df: DataFrame with dataset information
            val_ratio: Ratio of validation data
            
        Returns:
            Training and validation DataFrames
        """
        pass
    
    @abstractmethod
    def get_class_info(self, dataset_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about classes in the dataset.
        
        Args:
            dataset_df: DataFrame with dataset information
            
        Returns:
            Dictionary with class information
        """
        pass


class DataPartitionInterface(ABC):
    """Interface for dataset partitioning."""
    
    @abstractmethod
    def partition_dataset(self, data_dir: str, output_dir: str, 
                         server_ratio: float = 0.6,
                         client_ratio: float = 0.2) -> Dict[str, str]:
        """
        Partition a dataset for federated learning.
        
        Args:
            data_dir: Path to the dataset directory
            output_dir: Path to save the partitioned data
            server_ratio: Fraction of data for the server
            client_ratio: Fraction of data for each client
            
        Returns:
            Dictionary with paths to the partitioned data
        """
        pass
    
    @abstractmethod
    def create_unseen_classes(self, partitions: Dict[str, Any], 
                             data_dir: str, 
                             num_unseen_classes: int = 5) -> None:
        """
        Add unseen classes to client partitions.
        
        Args:
            partitions: Dictionary with partition information
            data_dir: Path to the dataset directory
            num_unseen_classes: Number of unseen classes to add
        """
        pass 