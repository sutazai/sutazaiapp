module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['<rootDir>/src/store/__tests__/**/*.(test|spec).js'],
  transform: {
    '^.+\\.[jt]sx?$': 'babel-jest',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleFileExtensions: ['js', 'jsx', 'json'],
};

