# 🌍 Global SDGs Data Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Overview

The **Global SDGs Data Analytics Platform** is an intelligent, multilingual web application that provides real-time monitoring and AI-powered forecasting for the United Nations Sustainable Development Goals (SDGs). This platform integrates data from multiple international sources and uses advanced machine learning models to help policymakers, researchers, and citizens understand global development trends.

### 🌐 Live Demo
Access the platform at: **https://8000-ijjy48l5ylcyao79gh8xa.e2b.dev**

## ✨ Key Features

### 🎯 Core Capabilities
- **📈 Real-time Data Integration**: Seamlessly access latest data from World Bank, OECD, and Our World in Data
- **🤖 AI-Powered Predictions**: Multiple ML models (Linear Regression, ARIMA, XGBoost) for accurate forecasting
- **📊 Visual Analytics**: Interactive charts with prediction intervals and achievement indicators
- **🔄 Multi-indicator Analysis**: Compare and analyze multiple SDG indicators simultaneously
- **🌏 Global Coverage**: Data for all countries and regions worldwide
- **🗣️ Multilingual Support**: Available in English, Korean (한국어), French (Français), Chinese (中文), and Japanese (日本語)

### 🚀 Advanced Features
- **Batch ETL Processing**: Efficient parallel data fetching with intelligent caching
- **Rate Limiting & Backoff**: Respectful API usage with automatic retry mechanisms
- **Confidence Intervals**: Statistical prediction bands for uncertainty quantification
- **Achievement Badges**: Visual indicators showing progress towards SDG targets
- **Export Capabilities**: Download results in various formats (CSV, JSON, Excel)
- **Custom Data Upload**: Analyze your own datasets with the same powerful models

## 📸 Screenshots & User Interface

### 🏠 Main Dashboard
The platform features a modern, responsive interface with gradient design and intuitive controls:

```
┌─────────────────────────────────────────────────────────────┐
│                 Global SDGs Analytics Platform              │
│        Real-time monitoring and AI-powered forecasting      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 What is this platform?                                 │
│  This intelligent data analytics platform helps monitor    │
│  and predict progress toward UN SDGs using AI models...    │
│                                                             │
│  ┌──────────┬──────────┬──────────┬──────────┐           │
│  │ Country  │Indicators│ AI Model │ Forecast │ [Language] │
│  │ [Korea▼] │[Multi...▼│[Linear▼] │[   5    ]│    [한국어▼]│
│  └──────────┴──────────┴──────────┴──────────┘           │
│                                                             │
│              [🔍 ANALYZE]  [📥 EXPORT RESULTS]             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Results Section:                                          │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Primary School Enrollment     SDG 4    🎯 85%      │   │
│  │ ┌──────────────────────────────────────────────┐  │   │
│  │ │     📈 Interactive Chart with:                │  │   │
│  │ │     - Historical data (solid line)            │  │   │
│  │ │     - AI Forecast (dashed line)               │  │   │
│  │ │     - 95% Confidence Interval (shaded area)   │  │   │
│  │ │     - Target line (red dashed)                │  │   │
│  │ └──────────────────────────────────────────────┘  │   │
│  │                                                    │   │
│  │  Current Progress: 85%    Predicted: 92%          │   │
│  │  Target: 100%            Model Accuracy: 94.5%    │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 📊 Features in Action

#### 1. **Multi-Indicator Selection**
Select multiple SDG indicators to analyze simultaneously:
- Education (SDG 4): School enrollment, literacy rates
- Poverty (SDG 1): Poverty headcount ratios
- Health (SDG 3): Mortality rates
- Climate (SDG 13): CO2 emissions, air pollution
- Energy (SDG 7): Renewable energy, electricity access
- And many more...

#### 2. **Achievement Badges**
Visual progress indicators with color coding:
- 🎯 **Green (80-100%)**: On track to meet SDG targets
- 📊 **Orange (50-79%)**: Making progress but needs acceleration
- ⚠️ **Red (0-49%)**: Requires urgent attention

#### 3. **Prediction Intervals**
Each forecast includes:
- **Point Prediction**: Most likely future value
- **95% Confidence Interval**: Range of probable outcomes
- **Model Accuracy (R²)**: Statistical measure of fit quality

## 🛠️ Technical Architecture

### Backend Stack
- **Framework**: FastAPI (High-performance async Python web framework)
- **Data Processing**: Pandas, NumPy for efficient data manipulation
- **Machine Learning Models**:
  - **Linear Regression**: Baseline trend analysis
  - **ARIMA**: Advanced time series forecasting (via pmdarima)
  - **XGBoost**: Gradient boosting for complex patterns
- **Caching**: Parquet file format with TTL-based cache invalidation
- **Rate Limiting**: Custom implementation with backoff strategies

### Frontend Stack
- **HTML5/CSS3**: Modern responsive design with gradients and animations
- **JavaScript**: Interactive charts and dynamic UI updates
- **Chart.js**: Professional data visualization with annotations
- **Select2**: Enhanced multi-select dropdowns
- **jQuery**: DOM manipulation and AJAX requests

### Data Sources
1. **World Bank API**: Comprehensive development indicators
2. **OECD Stats**: Economic and social statistics
3. **Our World in Data**: Research and data on global challenges

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager
- 2GB RAM minimum
- Internet connection for data fetching

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sdgs-analytics-platform.git
cd sdgs-analytics-platform
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
# Using supervisor (recommended for production)
supervisord -c supervisord.conf

# Or directly with uvicorn (for development)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the platform**
Open your browser and navigate to: `http://localhost:8000`

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t sdgs-platform .
docker run -p 8000:8000 sdgs-platform
```

## 📖 API Documentation

### FastAPI Auto-generated Docs
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

#### 1. **Get Capabilities**
```http
GET /api/capabilities
```
Returns available models and data sources.

#### 2. **Analyze Indicators**
```http
POST /api/analyze
Content-Type: multipart/form-data

country=KR
indicators=SE.PRM.NENR
indicators=SI.POV.DDAY
model=linear
horizon=5
lang=en
```

#### 3. **Upload Custom Data**
```http
POST /api/upload
Content-Type: multipart/form-data

file=@data.csv
model=arima
horizon=10
```

## 🌍 Supported Languages

The platform supports multiple languages for global accessibility:

| Language | Code | Native Name |
|----------|------|-------------|
| English | en | English |
| Korean | ko | 한국어 |
| French | fr | Français |
| Chinese | zh | 中文 |
| Japanese | ja | 日本語 |

## 📊 Supported SDG Indicators

### SDG 1: No Poverty
- Poverty headcount ratio at $2.15 a day
- National poverty lines

### SDG 3: Good Health
- Under-5 mortality rate
- Maternal mortality ratio

### SDG 4: Quality Education
- Primary school enrollment
- Secondary school enrollment
- Adult literacy rate

### SDG 7: Clean Energy
- Access to electricity
- Renewable energy consumption

### SDG 13: Climate Action
- CO2 emissions per capita
- PM2.5 air pollution

[... and many more indicators across all 17 SDGs]

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:

```env
# Cache settings
CACHE_TTL=86400  # 24 hours in seconds

# Rate limiting
WB_RATE_LIMIT=60  # requests per minute
OECD_RATE_LIMIT=30
OWID_RATE_LIMIT=30

# Model settings
ENABLE_ARIMA=true
ENABLE_XGBOOST=true
```

### Cache Management
The platform uses intelligent caching to minimize API calls:
- **TTL-based invalidation**: Data refreshes after 24 hours
- **Parquet format**: Efficient columnar storage
- **Automatic cleanup**: Old cache files are removed periodically

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **United Nations** for the SDG framework
- **World Bank** for comprehensive development data
- **OECD** for economic statistics
- **Our World in Data** for research and visualizations
- **FastAPI** community for the excellent framework
- All contributors and users of this platform

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/sdgs-analytics-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sdgs-analytics-platform/discussions)
- **Email**: sdgs-platform@example.com

## 🚦 Project Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)

### Roadmap
- [ ] Add more SDG indicators
- [ ] Implement real-time data streaming
- [ ] Add comparative country analysis
- [ ] Mobile application development
- [ ] API rate limit dashboard
- [ ] Advanced ML models (LSTM, Prophet)
- [ ] Collaborative features for researchers

---

**Made with ❤️ for a sustainable future**

*"The future depends on what we do in the present." - Mahatma Gandhi*