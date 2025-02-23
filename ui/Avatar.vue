<template>
  <div class="sutazai-avatar">
    <div class="hologram-effect"></div>
    <div class="facial-expression">
      <svg :class="expression">
        <!-- Animated SVG paths for facial features -->
      </svg>
    </div>
    <div class="particle-field"></div>
  </div>
</template>

<script>
export default {
  props: ['expression'],
  mounted() {
    this.initParticleField()
    this.startIdleAnimation()
  },
  methods: {
    initParticleField() {
      const particles = new SutazAiParticles({
        container: this.$el.querySelector('.particle-field'),
        count: 500,
        color: '#6366f1',
        connectionRadius: 150
      })
    },
    startIdleAnimation() {
      anime({
        targets: this.$el,
        translateY: [-10, 10],
        duration: 3000,
        easing: 'easeInOutQuad',
        loop: true
      })
    }
  }
}
</script> 