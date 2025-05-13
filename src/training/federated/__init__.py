"""
Federated learning training module.

This module contains components for training face recognition models
using federated learning approaches with multiple clients.

The federated learning process implemented here follows these steps:
1. Initialize a global model on the server
2. For each round:
   a. Clients download the global model
   b. Clients train the model on their local data
   c. Clients send model updates (not data) back to the server
   d. Server aggregates the updates to create a new global model
3. After all rounds, the final global model is saved

This approach preserves privacy by keeping all training data local to the clients
while still benefiting from the collective learning experience.
""" 