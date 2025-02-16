import { App } from '@slack/bolt';
const { WebClient } = require('@slack/web-api');
const { RateLimiter } = require('limiter');
require('dotenv').config();
const fs = require('fs');
const logger = require('./logger');
const { sanitize } = require('dompurify');
const aiService = require('./ai-service');
const helmet = require('helmet');
const rasp = require('node-rasp');
const crypto = require('crypto');
const { WasmAI } = require('wasm-ai-sdk');
const { constantTimeCompare } = require('crypto-secure');
const { create } = require('protobufjs-safe');
const root = await load("message.proto");
const Message = root.lookupType("secure.Message");
const axios = require('axios');
const { validateSlackRequest } = require('./security');
const Sentry = require('@sentry/node');
const circuitBreaker = require('opossum');
const CircuitBreaker = require('opossum');

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET
});

const token = process.env.SLACK_TOKEN;
const web = new WebClient(token);
const messageLimiter = new RateLimiter({ tokensPerInterval: 5, interval: 1000 });
const apiLimiter = new RateLimiter({ tokensPerInterval: 20, interval: "minute" });

const deadLetterQueue = [];
const alertMonitoringSystem = (message) => {
  logger.error('Critical failure:', message);
  web.chat.postMessage({
    channel: 'alerts',
    text: `Message failed: ${message.text}`
  });
};

class MessageQueue {
    constructor() {
        this.queue = [];
        this.MAX_RETRIES = 3;
    }

    addMessage(message, priority = 0) {
        this.queue.push({ ...message, priority, retries: 0 });
        this.queue.sort((a, b) => b.priority - a.priority);
    }

    async processQueue() {
        while (this.queue.length > 0) {
            const message = this.queue.shift();
            try {
                await sendMessage(message.channel, message.text);
            } catch (error) {
                if (message.retries < this.MAX_RETRIES) {
                    message.retries++;
                    this.addMessage(message, message.priority);
                } else {
                    logger.error('Message failed after maximum retries', { message });
                    deadLetterQueue.push(message);
                    alertMonitoringSystem(message);
                }
            }
        }
    }
}

const messageQueue = new MessageQueue();

// Slack bot logic here
app.message('hello', async ({ message, say }) => {
  await say(`Hello, <@${message.user}>!`);
});

const aiServiceBreaker = new CircuitBreaker(aiService.process, {
  timeout: 5000,
  errorThresholdPercentage: 50,
  resetTimeout: 60000
});

// Add AI response validation
const validateAIResponse = (response) => {
  if (typeof response !== 'string' || response.length > 2000) {
    throw new Error('Invalid AI response format');
  }
  if (/(<\s*(script|iframe)|javascript:)/i.test(response)) {
    throw new Error('XSS pattern detected');
  }
};

app.message(/^ai (.+)/i, async ({ context, say }) => {
  const rawInput = context.matches[1];
  const cleanInput = sanitize(rawInput, {ALLOWED_TAGS: []});
  
  if(cleanInput.length > 1000) {
    return await say("Input too long");
  }
  
  try {
    validateInput(cleanInput);
    const response = await aiServiceBreaker.fire(cleanInput);
    validateAIResponse(response);
    await say(`AI Response: ${response}`);
  } catch (error) {
    if (aiServiceBreaker.opened) {
      await say("AI service is currently overloaded. Please try again later.");
      logger.error(`AI circuit breaker open: ${error.stack}`);
    } else {
      await say(`Sorry, I encountered an error: ${error.message}`);
    }
  }
});

app.message(/^\/ask/, async ({ message, say }) => {
  try {
    if (rateLimitExceeded(message.user)) {
      return await say('Rate limit exceeded. Please wait 1 minute.');
    }
    
    const response = await getAIResponse(message.text);
    if (!response || response.trim() === '') {
      throw new Error('Empty AI response');
    }
    await say(response);
  } catch (error) {
    logger.error(`Error processing request: ${error}`);
    await say(`Sorry, I'm having trouble with that request. ${error.message}`);
  }
});

app.message(/secret/, async ({ message, say }) => {
  // Validate user permissions
  if (!isAuthorized(message.user)) {
    logger.warn(`Unauthorized access attempt by ${message.user}`);
    return;
  }
  
  // Parameterized query to prevent SQL injection
  const response = await db.query('SELECT * FROM data WHERE content = $1', [message.text]);
});

const deploy = async () => {
  await app.start(process.env.PORT || 3000);
  logger.info('Slack Bot is running!');
};

if (process.argv.includes('--deploy')) {
  deploy();
}

// Add input validation
function validateInput(input) {
  if (!input || typeof input !== 'string') {
    throw new Error('Invalid input');
  }
  // Add more validation as needed
}

// Add error handling
app.error((error) => {
  logger.error('Slack bot error:', error);
  // Implement proper error reporting
});

// Add rate limiting
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per window
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);

app.use(helmet.contentSecurityPolicy({
    directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'"]
    }
}));

async function sendMessage(channel, text) {
    try {
        await messageLimiter.removeTokens(1);
        if (!channel || !text) {
            throw new Error('Invalid input parameters');
        }
        const result = await web.chat.postMessage({
            channel: channel,
            text: text
        });
        return result;
    } catch (error) {
        logger.error('Error sending message:', error);
        throw error;
    }
}

// Add message persistence
function saveMessageQueue() {
    fs.writeFileSync('message_queue.json', JSON.stringify(messageQueue.queue));
}

function loadMessageQueue() {
    if (fs.existsSync('message_queue.json')) {
        messageQueue.queue = JSON.parse(fs.readFileSync('message_queue.json'));
    }
}

// Load messages on startup
loadMessageQueue();

class SlackBot {
  constructor(token) {
    if (!token) {
      throw new Error('Slack token is required');
    }
    this.client = new WebClient(token);
    this.rateLimit = {
      remaining: 1,
      reset: 0
    };
  }

  async sendMessage(channel, message) {
    try {
      if (!channel || !message) {
        throw new Error('Channel and message are required');
      }
      
      // Check rate limit
      const now = Date.now() / 1000;
      if (this.rateLimit.remaining <= 0 && now < this.rateLimit.reset) {
        throw new Error('Rate limit exceeded');
      }

      const response = await this.client.chat.postMessage({
        channel,
        text: message
      });
      
      // Update rate limit
      this.rateLimit = {
        remaining: response.headers['x-ratelimit-remaining'],
        reset: response.headers['x-ratelimit-reset']
      };
      
      return response;
    } catch (error) {
      logger.error(`Failed to send message: ${error.message}`);
      throw error;
    }
  }
}

app.post('/message', async (req, res) => {
  try {
    // Added input validation
    if (!req.body.text || req.body.text.length > 500) {
      return res.status(400).send('Invalid message');
    }
    
    // Added rate limiting
    const rateLimit = await checkRateLimit(req.ip);
    if (rateLimit.exceeded) {
      return res.status(429).send('Too many requests');
    }
    
    // Add input sanitization
    const cleanText = req.body.text.replace(/[<>]/g, '');
    
    await postMessage(req.body);
  } catch (error) {
    logger.error('Error:', error);
    res.status(500).send('Internal error');
  }
});

// Add global error handler
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Send to error monitoring service
});

// Wrap API calls in try/catch
async function handleMessage(event) {
  try {
    await apiLimiter.removeTokens(1);
    // Existing message handling
  } catch (error) {
    logger.error(`Error processing message: ${error.message}`);
    await retryWithBackoff(() => sendErrorMessage(event.channel));
  }
}

// Add RASP protection
rasp.init({
  antiDebug: true,
  memoryProtection: true,
  requestInspection: true
});

// Add behavioral monitoring
const userBehavior = new Map();
app.message(async ({ message, client, say }) => {
  try {
    // Added input validation
    if (!message.text || message.subtype) return;
    
    // Added rate limiting
    const rateLimit = checkRateLimit(message.user);
    if (rateLimit.exceeded) {
      await say(`Slow down! Try again in ${rateLimit.remaining} seconds`);
      return;
    }
    
    // Sanitize user input
    const sanitizedInput = sanitize(message.text);
    await processRequest(sanitizedInput);
    
  } catch (error) {
    logger.error(`Error processing message: ${error.stack}`);
    // Added error recovery
    await say("Oops! Something went wrong. Our team has been notified.");
  }
});

// Add memory-safe AI processing
const aiModel = new WasmAI('secure_model.wasm');

// Add request fingerprinting
const fingerprintMiddleware = async ({ body, next }) => {
    const fingerprint = crypto
        .createHash('sha256')
        .update(body.rawRequest.toString())
        .digest('hex');
        
    if(await checkRevoked(fingerprint)) {
        throw new Error('Request fingerprint revoked');
    }
    await next();
};

// Security middleware sequence
app.use(validateRequestSize);    // 1. Size validation (1MB limit)
app.use(verifySignature);        // 2. HMAC verification
app.use(validateProtocolBuffer); // 3. Protobuf validation
app.use(fingerprintMiddleware);  // 4. Request fingerprinting

function verifySignature(message, signature) {
    const expected = crypto
        .createHmac('sha512', process.env.SECRET)
        .update(message)
        .digest('hex');
    return constantTimeCompare(signature, expected);
}

// Add request body validation
app.use(async ({ body, next }) => {
    try {
        Message.verify(body.raw);
        await next();
    } catch (e) {
        throw new Error(`Invalid protocol buffer: ${e.message}`);
    }
});

app.command('/deploy', async ({ ack, body, say }) => {
  try {
    await ack();
    const user = body.user_id;
    if (!isAuthorized(user)) {
      logger.warn(`Unauthorized deploy attempt by ${user}`);
      throw new Error('Unauthorized deployment attempt');
    }
    const result = await Promise.race([
      deployAll(),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Deployment timeout')), 5000)
      )
    ]);
    await say(`Deployment started: ${result}`);
  } catch (error) {
    Sentry.captureException(error);
    logger.error(`Deployment error: ${error.stack}`);
    await say(`Error: ${error.message}`);
  }
});

function checkWorkflowStatus(workflowId) {
  return axios.get(`https://api.github.com/repos/your/repo/actions/runs/${workflowId}`, {
    headers: { Authorization: `token ${process.env.GITHUB_TOKEN}` }
  });
}

// Add request validation middleware
app.use(async ({ body, next }) => {
    if (!validateRequestSignature(body)) {
        throw new Error('Invalid request signature');
    }
    await next();
});

function validateRequestSignature(body) {
  return constantTimeCompare(
    crypto.createHmac('sha256', process.env.SIGNING_SECRET)
      .update(body.rawRequest.toString())
      .digest('hex'),
    body.headers['x-slack-signature']
  );
}

// Add request validation
app.use(async ({ body, next }) => {
    if (!body.rawRequest || typeof body.rawRequest !== 'string') {
        throw new Error('Invalid request format');
    }
    await next();
});

// Add middleware ordering
app.use(async ({ body, next }) => {
    // Request size validation first
    if (body.rawRequest.length > 1_000_000) {
        throw new Error('Request payload too large');
    }
    await next();
});

// Reorder security middleware
app.use(validateRequestSize);  // 1. Size check
app.use(verifySignature);     // 2. Authentication
app.use(validateProtocolBuffer); // 3. Data validation

// Add request schema validation
app.use(validateRequestSchema(schema));

// Add middleware ordering
app.use(async ({ body, next }) => {
    // Request size validation first
    if (body.rawRequest.length > 1_000_000) {
        throw new Error('Request payload too large');
    }
    await next();
});

// Reorder security middleware
app.use(validateRequestSize);  // 1. Size check
app.use(verifySignature);     // 2. Authentication
app.use(validateProtocolBuffer); // 3. Data validation 

function sendAlert(message) {
  const webhookUrl = process.env.SLACK_WEBHOOK;
  axios.post(webhookUrl, { text: message })
    .catch(error => {
      logger.error('Slack notification failed', { error });
      // Add retry logic
      setTimeout(() => axios.post(webhookUrl, { text: message }), 5000);
    });
}

app.post('/slack/events', async (req, res) => {
  try {
    if (!validateSlackRequest(req)) {
      return res.status(403).send('Invalid request');
    }
    // Existing logic...
  } catch (error) {
    console.error(`Slack bot error: ${error.stack}`);
    Sentry.captureException(error);
    res.status(500).send('Internal server error');
  }
});

// Add request size validation middleware
app.use(async ({ body, next }) => {
    const MAX_SIZE = 1024 * 1024; // 1MB
    if (body.rawRequest.length > MAX_SIZE) {
        logger.warn(`Oversized request from ${body.ip} (${body.rawRequest.length} bytes)`);
        throw new Error('Request payload exceeds 1MB limit');
    }
    await next();
});

// Add rate limiting to message handler
const userRateLimits = new Map();
app.message(async ({ message }) => {
    const userId = message.user;
    const limit = userRateLimits.get(userId) || { count: 0, last: Date.now() };
    
    if (Date.now() - limit.last < 1000 && limit.count >= 5) {
        throw new Error('Rate limit exceeded');
    }
    
    limit.count = Date.now() - limit.last < 1000 ? limit.count + 1 : 1;
    limit.last = Date.now();
    userRateLimits.set(userId, limit);
}); 