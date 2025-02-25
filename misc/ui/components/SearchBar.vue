<template>
  <v-text-field
    v-model="searchQuery"
    label="Search"
    prepend-icon="mdi-magnify"
    @input="search"
  ></v-text-field>
</template>

<script>
export default {
  data() {
    return {
      searchQuery: ''
    }
  },
  methods: {
    search() {
      this.$emit('search', this.searchQuery)
    }
  }
}
</script> 