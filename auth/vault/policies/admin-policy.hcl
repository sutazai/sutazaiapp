# Vault policy for administrators
path "*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

path "sys/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

path "auth/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}