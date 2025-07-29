# pHdockUI Website

This is the website for the pH-aware Molecular Docking Suite, designed to showcase our research capabilities to academic researchers at top institutions.

## 🚀 Quick Start

### Option 1: Using the Run Script (Recommended)
```bash
./run-local.sh
```
Then choose option 3 for a quick frontend-only demo.

### Option 2: Manual Frontend Only
```bash
npm install
npm run dev
```
Visit http://localhost:3000

### Option 3: Full Stack with Docker Compose
```bash
docker-compose up
```
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 4: Full Stack Manual Setup

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Terminal 2 - Frontend:**
```bash
npm install
npm run dev
```

## 📁 Project Structure

```
website/
├── app/                    # Next.js pages and layouts
│   ├── page.tsx           # Home page with interactive interface
│   ├── docs/              # Documentation page
│   ├── about/             # About page
│   └── contact/           # Contact page
├── components/            # React components
│   ├── Navigation.tsx     # Top navigation bar
│   ├── MoleculeInterface.tsx  # Main interactive demo
│   ├── ResultsPanel.tsx   # Results display
│   └── ...
├── backend/               # FastAPI backend
│   ├── main.py           # API endpoints
│   └── requirements.txt   # Python dependencies
├── public/                # Static assets
└── docker-compose.yml     # Docker configuration
```

## 🎯 Key Features

1. **Interactive Demo**: Paste SMILES or upload SDF files to run pH-aware docking analysis
2. **Real-time Results**: View pKa predictions, protonation states, and docking scores
3. **Dark Mode**: Toggle between light and dark themes
4. **Responsive Design**: Works on desktop and mobile devices
5. **Academic Focus**: Clean, professional design suitable for research presentations

## 🔧 Environment Variables

Create a `.env.local` file in the root directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🧪 Testing Locally

1. **Test the Interface**: 
   - Click on example molecules (Aspirin, Ibuprofen, etc.)
   - Try uploading an SDF file
   - Adjust pH values and other parameters

2. **Check Responsiveness**:
   - Resize browser window
   - Test on mobile device using DevTools

3. **Dark Mode**:
   - Click the sun/moon icon in navigation

## 📝 Customization

### Update Team Information
Edit `/app/about/page.tsx` to add your team members' names and bios.

### Update Contact Information
Edit `/app/contact/page.tsx` to add your email and institution.

### Update GitHub Links
Search for "yourusername" and replace with your actual GitHub username.

### Update Citations
Edit `/components/CredibilitySection.tsx` to add your publications.

## 🚢 Deployment

### Option 1: Vercel (Recommended for Frontend)
1. Push to GitHub
2. Import project on Vercel
3. Set environment variables
4. Deploy

### Option 2: Docker on VPS
1. Build images: `docker-compose build`
2. Run: `docker-compose up -d`
3. Set up reverse proxy (nginx/caddy)

### Option 3: University Infrastructure
Contact your IT department for deployment on institutional servers.

## 🐛 Troubleshooting

**Frontend won't start:**
- Delete `node_modules` and `.next` folders
- Run `npm install` again

**Backend connection errors:**
- Check if backend is running on port 8000
- Verify NEXT_PUBLIC_API_URL is set correctly

**Docker issues:**
- Ensure Docker daemon is running
- Check port conflicts (3000, 8000, 6379)

## 📧 Support

For issues or questions, please open an issue on GitHub or contact the team through the website's contact form.

## 🔒 Security Notes

- The current setup uses in-memory job storage (not suitable for production)
- Add proper authentication before deploying publicly
- Configure CORS appropriately for your domain
- Use HTTPS in production

## 📄 License

This project is part of the pHdockUI suite and follows the same license terms. 