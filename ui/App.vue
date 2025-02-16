<template>
  <v-app>
    <v-navigation-drawer v-model="drawer" app>
      <v-list>
        <v-list-item>
          <v-list-item-content>
            <v-list-item-title class="title">
              SutazAI
            </v-list-item-title>
            <v-list-item-subtitle>
              Infinite Possibilities
            </v-list-item-subtitle>
          </v-list-item-content>
        </v-list-item>
        <v-divider></v-divider>
        <v-list-item
          v-for="item in menuItems"
          :key="item.title"
          link
          @click="navigate(item.route)"
        >
          <v-list-item-icon>
            <v-icon>{{ item.icon }}</v-icon>
          </v-list-item-icon>
          <v-list-item-content>
            <v-list-item-title>{{ item.title }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <v-app-bar app color="primary" dark>
      <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      <v-toolbar-title>SutazAI Dashboard</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn icon @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
    </v-app-bar>

    <v-main>
      <router-view></router-view>
      <NotificationSystem />
      <v-container>
        <v-btn @click="toggleHighContrast">Toggle High Contrast</v-btn>
        <v-btn @click="increaseFontSize">Increase Font Size</v-btn>
        <v-btn @click="decreaseFontSize">Decrease Font Size</v-btn>
      </v-container>
    </v-main>
  </v-app>
</template>

<script>
import NotificationSystem from '@/components/NotificationSystem.vue';

export default {
  components: {
    NotificationSystem
  },
  data() {
    return {
      drawer: true,
      menuItems: [
        { title: 'Dashboard', icon: 'mdi-view-dashboard', route: '/' },
        { title: 'Terminal', icon: 'mdi-console', route: '/terminal' },
        { title: 'Settings', icon: 'mdi-cog', route: '/settings' }
      ]
    }
  },
  methods: {
    navigate(route) {
      this.$router.push(route)
    },
    toggleTheme() {
      this.$vuetify.theme.dark = !this.$vuetify.theme.dark
    },
    toggleHighContrast() {
      document.body.classList.toggle('high-contrast')
    },
    increaseFontSize() {
      const currentSize = parseFloat(getComputedStyle(document.body).fontSize)
      document.body.style.fontSize = `${currentSize + 2}px`
    },
    decreaseFontSize() {
      const currentSize = parseFloat(getComputedStyle(document.body).fontSize)
      document.body.style.fontSize = `${currentSize - 2}px`
    }
  }
}
</script>

<style>
.v-application {
  background-color: var(--v-background-base);
}

.high-contrast {
  background-color: black !important;
  color: white !important;
}
</style> 