# Development Guide

## SPA Routing Setup

This application uses React Router for client-side routing. The configuration ensures that refreshing the page on any route works correctly.

### Configuration Details

1. **Vite Configuration** (`vite.config.ts`):
   - `appType: 'spa'` - Tells Vite this is a Single Page Application
   - Fallback to `index.html` for all routes during development and preview

2. **React Router** (`src/routes.tsx`):
   - Catch-all route (`path="*"`) handles unknown routes with a 404 page
   - All routes are defined with proper components

### Common Issues and Solutions

#### Issue: "Page Not Found" on Refresh

**Symptoms:**
- Page works when navigating via links
- Refreshing the browser shows "404 Not Found" or "Cannot GET /profile/123"

**Solutions:**

1. **Development Server (Vite):**
   ```bash
   npm run dev
   ```
   Should work automatically with the current configuration.

2. **Production Build:**
   ```bash
   npm run build
   npm run preview
   ```
   The preview server should handle SPA routing correctly.

3. **Custom Server Deployment:**
   If deploying to a custom server, ensure it's configured to serve `index.html` for all routes:
   
   - **Nginx:**
     ```nginx
     location / {
       try_files $uri $uri/ /index.html;
     }
     ```
   
   - **Apache (.htaccess):**
     ```apache
     RewriteEngine On
     RewriteBase /
     RewriteRule ^index\.html$ - [L]
     RewriteCond %{REQUEST_FILENAME} !-f
     RewriteCond %{REQUEST_FILENAME} !-d
     RewriteRule . /index.html [L]
     ```

4. **Vercel/Netlify:**
   These platforms handle SPA routing automatically, but you can add a `_redirects` file or `vercel.json` for explicit configuration.

### Testing SPA Routing

1. Start the development server:
   ```bash
   npm run dev
   ```

2. Navigate to any route (e.g., `http://localhost:5174/profile/123`)

3. Refresh the page - it should still show the correct page, not a 404

4. Test the catch-all route by visiting a non-existent route (e.g., `http://localhost:5174/nonexistent`)

### Environment Variables

Make sure your `.env` file is properly configured:

```bash
# Copy the example file
cp env.example .env

# Edit the values as needed
```

### Troubleshooting

If you're still experiencing routing issues:

1. **Clear browser cache** and hard refresh (Ctrl+Shift+R)

2. **Check the console** for any JavaScript errors

3. **Verify environment variables** are loaded correctly

4. **Restart the development server** after configuration changes

5. **Check network tab** to see what requests are being made when refreshing 