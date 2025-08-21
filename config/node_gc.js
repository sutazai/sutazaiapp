// Node.js Garbage Collection Configuration
process.env.NODE_OPTIONS = [
    '--max-old-space-size=256',  // Limit heap size to 256MB
    '--max-semi-space-size=64',  // Limit young generation
    '--optimize-for-size',       // Optimize for memory usage
    '--gc-interval=100'          // Force GC more frequently
].join(' ');

// Monitor memory usage
setInterval(() => {
    const usage = process.memoryUsage();
    if (usage.heapUsed > 200 * 1024 * 1024) { // 200MB threshold
        if (global.gc) {
            global.gc();
            console.log('Forced garbage collection triggered');
        }
    }
}, 30000); // Check every 30 seconds
