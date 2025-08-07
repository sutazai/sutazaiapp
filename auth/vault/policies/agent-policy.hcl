# Vault policy for AI agents
path "secret/data/service-accounts/*" {
  capabilities = ["read"]
}

path "secret/data/shared/jwt-secret" {
  capabilities = ["read"]
}

path "secret/metadata/service-accounts/*" {
  capabilities = ["list", "read"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

path "sys/leases/renew" {
  capabilities = ["update"]
}