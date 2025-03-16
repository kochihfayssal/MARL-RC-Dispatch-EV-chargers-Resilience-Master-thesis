import torch
torch.cuda.empty_cache()  # If you're using a GPU

print(torch.cuda.is_available())

# # The function we want to test
# def normalize_input(input_tensor, dimension=-1, epsilon=1e-5):
#     mean = input_tensor.mean(dim=dimension, keepdim=True)
#     var = input_tensor.var(dim=dimension, unbiased=False, keepdim=True)
#     var = var.clamp(min=epsilon)
#     normalized_observation = (input_tensor - mean) / torch.sqrt(var)
#     return normalized_observation

# # Test function
# def test_normalize_input():
#     # Create a random tensor of shape (3, 5)
#     input_tensor = torch.randn(3, 3, 5)  
    
#     # Apply the normalization function
#     normalized_tensor = normalize_input(input_tensor)

#     print(input_tensor, normalized_tensor, sep="\n")

#     # Calculate mean and variance of the normalized tensor
#     mean = normalized_tensor.mean(dim=-1)
#     var = normalized_tensor.var(dim=-1, unbiased=False)
    
#     # Check if the mean is approximately 0
#     assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6), "Mean is not close to 0"
    
#     # Check if the variance is approximately 1
#     assert torch.allclose(var, torch.ones_like(var), atol=1e-6), "Variance is not close to 1"
    
#     print("Test passed: Mean is 0 and variance is 1.")

# # Run the test
# test_normalize_input()
