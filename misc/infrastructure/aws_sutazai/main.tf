resource "aws_instance" "compute_node" {
  tags = { Name = "sutazai-compute" }
} 