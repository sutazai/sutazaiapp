const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('<html><head><title>My Test App</title></head><body><h1>Hello from Express</h1></body></html>');
});

app.listen(3000, () => {
  console.log('Server listening on http://localhost:3000');
});
