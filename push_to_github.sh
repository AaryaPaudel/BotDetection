#!/bin/bash
# Push to GitHub after creating the repository

echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✓ Successfully pushed to GitHub!"
    echo "Repository: https://github.com/AaryaPaudel/BotDetection"
else
    echo "✗ Push failed. Make sure you've created the repository on GitHub first."
    echo ""
    echo "To create the repository:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: BotDetection"
    echo "3. Description: Fake Review Detection System - AI Coursework CU6051NI"
    echo "4. Choose Public or Private"
    echo "5. DO NOT initialize with README, .gitignore, or license"
    echo "6. Click 'Create repository'"
    echo "7. Then run this script again: ./push_to_github.sh"
fi

