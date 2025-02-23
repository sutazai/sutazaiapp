<template>
  <div v-if="error" class="error">
    {{ error }}
  </div>
  <div v-else>
    <v-textarea v-model="output" readonly />
    <v-text-field v-model="command" @keyup.enter="executeCommand" />
  </div>
</template>

<script>
export default {
  data() {
    return {
      output: "",
      command: "",
      error: ""
    };
  },
  methods: {
    async executeCommand() {
      try {
        const result = await this.$api.post("/execute", { command: this.command });
        this.output += `$ ${this.command}\n${result.data}\n`;
        this.command = "";
      } catch (e) {
        this.error = "Failed to execute command";
      }
    }
  }
};
</script>