<template>
  <div v-if="error" class="error">
    {{ error }}
  </div>
  <div v-else>
    {{ address }}
  </div>
</template>

<script>
// Consolidate address formatting
const addressFormats = {
  'US': (address) => `${address.street}, ${address.city}, ${address.state} ${address.zip}`,
  'UK': (address) => `${address.street}, ${address.city}, ${address.postcode}`,
  'default': (address) => `${address.street}, ${address.city}, ${address.postal_code}`
};

export default {
  data() {
    return {
      address: "",
      error: ""
    };
  },
  methods: {
    async fetchAddress() {
      try {
        const response = await this.$api.get("/address");
        self.address = response.data;
      } catch (e) {
        self.error = "Failed to fetch address";
      }
    },
    formatAddress(address) {
      const formatter = addressFormats[address.country] || addressFormats['default'];
      return formatter(address);
    }
  },
  mounted() {
    self.fetchAddress();
  }
};
</script>

<style scoped>
.v-card {
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.v-btn {
  text-transform: none;
  letter-spacing: normal;
}
</style> 