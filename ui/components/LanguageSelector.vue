<template>
  <v-select
    v-model="selectedLanguage"
    :items="languages"
    label="Language"
    @change="changeLanguage"
  ></v-select>
</template>

<script>
export default {
  data() {
    return {
      selectedLanguage: 'en',
      languages: [
        { text: 'English', value: 'en' },
        { text: 'Spanish', value: 'es' },
        { text: 'French', value: 'fr' }
      ]
    }
  },
  methods: {
    changeLanguage() {
      this.$i18n.locale = this.selectedLanguage
    }
  }
}
</script> 