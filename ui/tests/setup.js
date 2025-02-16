import { config } from '@vue/test-utils'
import { expect } from 'vitest'
import matchers from '@testing-library/jest-dom/matchers'

// Add jest-dom matchers
expect.extend(matchers)

// Global Vue Test Utils configuration
config.global.mocks = {
  $t: (msg) => msg
}

config.global.stubs = {
  'router-link': true,
  'router-view': true
} 