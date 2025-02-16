import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import App from '../src/App.vue'

describe('App.vue', () => {
  it('renders properly', () => {
    const wrapper = mount(App)
    expect(wrapper.text()).toContain('SutazAi')
  })

  it('has correct theme classes', () => {
    const wrapper = mount(App)
    expect(wrapper.find('div').classes()).toContain('bg-background')
    expect(wrapper.find('div').classes()).toContain('text-text')
  })

  it('contains navigation and notifications', () => {
    const wrapper = mount(App)
    expect(wrapper.findComponent('NavBar').exists()).toBe(true)
    expect(wrapper.findComponent('Notifications').exists()).toBe(true)
  })
}) 