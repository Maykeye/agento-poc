#!/usr/bin/env node
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const TIMEOUT_MS = 1000; // 1 second total
async function main() {
    const htmlPath = process.argv[2] || './index.html';

    // Validate file exists
    if (!fs.existsSync(htmlPath)) {
        console.error(`❌ File not found: ${htmlPath}`);
        process.exit(1);
    }

    const filePath = path.resolve(htmlPath);
    const url = `file://${filePath.replace(/\\/g, '/')}`;

    console.log(`🚀 Loading: ${url}\n`);

    // Launch browser in headless mode
    const browser = await puppeteer.launch({
        headless: 'new',
        //args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
        const page = await browser.newPage();

        // Capture all console messages and send to stdout
        page.on('console', (msg) => {
            const type = msg.type();
            const text = msg.text().replace(/\n/g, '\n      ');
            console.log(`[${type}] ${text}`);
        });

        // Capture JavaScript errors that weren't caught
        page.on('pageerror', (error) => {
            console.error(`[PAGE ERROR] ${error.message}\n${error.stack.replace(/\n/g, '\n          ')}`);
        });

        // Load the HTML file and wait for it to fully load
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS  });

    } catch (error) {
        console.error(`[BROWSER ERROR] ${error.message}`);
        process.exit(1);
    } finally {
        await browser.close();
    }

    console.log('\n✅ Done');
}

main().catch(err => {
    console.error('Fatal:', err);
    process.exit(1);
});
