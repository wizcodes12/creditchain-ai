import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Model Loading Paths (Hardcoded as per request) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDIT_MODEL_PATH = os.path.join(BASE_DIR, "ai_ml", "credit_scoring_model", "credit_scoring_model.h5")
CREDIT_PREPROCESSORS_PATH = os.path.join(BASE_DIR, "ai_ml", "credit_scoring_model", "credit_scoring_preprocessors.pkl")
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "ai_ml", "anomaly_detection_model", "anomaly_detection_model.h5")
ANOMALY_PREPROCESSORS_PATH = os.path.join(BASE_DIR, "ai_ml", "anomaly_detection_model", "anomaly_detection_preprocessors.pkl")

# --- Load Models and Preprocessors ---
def load_model_and_preprocessors():
    try:
        # Load Credit Scoring Model
        credit_model = tf.keras.models.load_model(CREDIT_MODEL_PATH, compile=False)
        with open(CREDIT_PREPROCESSORS_PATH, 'rb') as f:
            credit_preprocessors = pickle.load(f)
        
        # Load Anomaly Detection Model
        # Custom objects for anomaly model if needed (e.g., custom loss function)
        def reconstruction_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        anomaly_model = tf.keras.models.load_model(
            ANOMALY_MODEL_PATH, 
            custom_objects={'reconstruction_loss': reconstruction_loss}, 
            compile=False
        )
        with open(ANOMALY_PREPROCESSORS_PATH, 'rb') as f:
            anomaly_preprocessors = pickle.load(f)

        return credit_model, credit_preprocessors, anomaly_model, anomaly_preprocessors
    except Exception as e:
        raise RuntimeError(f"Failed to load models or preprocessors: {e}")

credit_model, credit_preprocessors, anomaly_model, anomaly_preprocessors = load_model_and_preprocessors()

# Extract components for Credit Scoring
credit_scaler = credit_preprocessors['scaler']
credit_encoders = credit_preprocessors['encoders']
credit_feature_columns = credit_preprocessors['feature_columns']

# Extract components for Anomaly Detection
anomaly_scaler = anomaly_preprocessors['scaler']
anomaly_encoders = anomaly_preprocessors['encoders']
anomaly_feature_columns = anomaly_preprocessors['feature_columns']
anomaly_iso_forest = anomaly_preprocessors['iso_forest']

# --- Define FastAPI App ---
app = FastAPI(
    title="CreditChain AI ML Service",
    description="API for Credit Scoring and Anomaly Detection Models",
    version="1.0.0"
)

# --- Pydantic Schemas for API Input/Output ---

# Anomaly Detection Model Input/Output
class LocationSchema(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    city: str
    state: str

class TransactionDataInput(BaseModel):
    transactionId: str
    amount: float = Field(..., ge=0.01, le=1000000)
    type: str = Field(..., pattern="^(debit|credit)$")
    category: str = Field(..., pattern="^(groceries|dining|entertainment|transportation|utilities|healthcare|education|shopping|travel|fuel|insurance|loan_payment|salary|investment|business|other)$")
    subcategory: Optional[str] = None
    date: str # Using string for date as per schema, will parse if needed
    hourOfDay: int = Field(..., ge=0, le=23)
    dayOfWeek: int = Field(..., ge=0, le=6)
    isWeekend: bool
    isLateNight: bool
    merchant: str
    location: LocationSchema
    paymentMethod: Optional[str] = Field(None, pattern="^(card|upi|netbanking|cash|wallet)$")
    balanceAfterTransaction: Optional[float] = None

class ContextualFeaturesInput(BaseModel):
    timeSinceLastTransaction: float
    avgAmountForCategoryLast7Days: float
    stdDevAmountForCategoryLast30Days: float
    userAvgTransactionAmount: float
    userStdDevTransactionAmount: float
    locationDeviationFromUsualPatterns: float
    merchantFrequencyForUser: float
    consecutiveTransactionsSameMerchant: int
    velocityOfTransactionsInShortPeriod: int
    unusualTimePattern: bool
    geographicalOutlier: bool
    amountOutlierForCategory: bool
    merchantOutlier: bool

class UserBehavioralContextInput(BaseModel):
    avgMonthlySpending: float
    typicalTransactionCountPerDay: int
    historicalAnomalyRateForUser: float
    averageTransactionAmountLast30Days: float
    mostCommonCategories: List[str]
    mostCommonMerchants: List[str]
    usualTransactionHours: List[int]
    typicalGeographicalArea: Dict[str, Any] # Using Dict[str, Any] for nested object

class AnomalyDetectionModelInput(BaseModel):
    transactionData: TransactionDataInput
    contextualFeatures: ContextualFeaturesInput
    userBehavioralContext: UserBehavioralContextInput

class RiskFactorOutput(BaseModel):
    factor: str
    weight: float = Field(..., ge=0, le=100)
    description: Optional[str] = None

class AnomalyReasonOutput(BaseModel):
    reason: str
    severity: str = Field(..., pattern="^(low|medium|high)$")
    confidence: int = Field(..., ge=0, le=100)

class AnomalyDetectionModelOutput(BaseModel):
    transactionId: str
    isAnomaly: bool
    anomalyScore: float = Field(..., ge=0, le=100)
    fraudRisk: str = Field(..., pattern="^(low|medium|high|critical)$")
    detectedPatterns: List[str]
    riskFactors: List[RiskFactorOutput]
    anomalyReasons: List[AnomalyReasonOutput]
    reconstructionError: float
    ensembleScore: float

# Credit Scoring Model Input/Output
class PersonalInfoInput(BaseModel):
    age: int = Field(..., ge=18, le=80)
    monthlyIncome: float = Field(..., ge=15000, le=500000)
    monthlyExpenses: float = Field(..., ge=10000, le=400000)
    employmentType: str = Field(..., pattern="^(salaried|self_employed|business_owner|freelancer)$")
    experienceYears: int = Field(..., ge=0, le=50)
    companyName: str

class ExistingLoansInput(BaseModel):
    totalAmount: float = Field(..., ge=0, le=10000000)
    monthlyEMI: float = Field(..., ge=0, le=100000)
    loanTypes: List[str] = Field(..., min_length=0)
    activeLoanCount: int = Field(..., ge=0, le=10)

class CreditCardsInput(BaseModel):
    totalLimit: float = Field(..., ge=0, le=1000000)
    currentUtilization: float = Field(..., ge=0, le=100)
    utilizationAmount: float = Field(..., ge=0)
    cardCount: int = Field(..., ge=0, le=10)

class BankAccountsInput(BaseModel):
    accountType: str = Field(..., pattern="^(savings|current|salary)$")
    averageBalance: float = Field(..., ge=1000, le=5000000)
    accountAge: int = Field(..., ge=1, le=30)

class AlternativeFactorsInput(BaseModel):
    digitalPaymentScore: float = Field(..., ge=0, le=100)
    socialMediaScore: float = Field(..., ge=0, le=100)
    appUsageScore: float = Field(..., ge=0, le=100)
    locationStabilityScore: float = Field(..., ge=0, le=100)
    phoneUsagePattern: float = Field(..., ge=0, le=100)

class BehavioralPatternsInput(BaseModel):
    paymentRegularity: float = Field(..., ge=0, le=100)
    transactionFrequency: float = Field(..., ge=0, le=100)
    financialDiscipline: float = Field(..., ge=0, le=100)
    riskTolerance: float = Field(..., ge=0, le=100)

class CalculatedRatiosInput(BaseModel):
    incomeToExpenseRatio: float = Field(..., ge=0)
    debtToIncomeRatio: float = Field(..., ge=0, le=100)
    creditUtilizationRatio: float = Field(..., ge=0, le=100)

class UserProfileData(BaseModel):
    age: int
    monthlyIncome: float
    monthlyExpenses: float
    employmentType: str
    experienceYears: int
    companyName: str
    existingLoans_totalAmount: float
    existingLoans_monthlyEMI: float
    existingLoans_loanTypes: List[str]
    existingLoans_activeLoanCount: int
    creditCards_totalLimit: float
    creditCards_currentUtilization: float
    creditCards_utilizationAmount: float
    creditCards_cardCount: int
    bankAccounts_accountType: str
    bankAccounts_averageBalance: float
    bankAccounts_accountAge: int
    digitalPaymentScore: float
    socialMediaScore: float
    appUsageScore: float
    locationStabilityScore: float
    phoneUsagePattern: float
    paymentRegularity: float
    transactionFrequency: float
    financialDiscipline: float
    riskTolerance: float
    incomeToExpenseRatio: float
    debtToIncomeRatio: float
    creditUtilizationRatio: float

class TransactionMetricsInput(BaseModel):
    totalTransactionsCount: int
    avgTransactionAmount: float
    maxTransactionAmount: float
    minTransactionAmount: float
    numCreditTransactions: int
    numDebitTransactions: int
    avgDailyTransactions: float
    avgMonthlySpendingGroceries: float
    avgMonthlySpendingDining: float
    avgMonthlySpendingEntertainment: float
    avgMonthlySpendingTransportation: float
    avgMonthlySpendingUtilities: float
    avgMonthlySpendingHealthcare: float
    avgMonthlySpendingEducation: float
    avgMonthlySpendingShopping: float
    avgMonthlySpendingTravel: float
    avgMonthlySpendingFuel: float
    avgMonthlySpendingInsurance: float
    avgMonthlySpendingLoanPayment: float
    percentageSpendingNeedsVsWants: float
    spreadOfTransactionsAcrossCategories: float
    loanRepaymentConsistency: float
    creditCardBillPaymentRegularity: float
    numberOfUniqueMerchants: int
    weekendSpendingPattern: float
    lateNightTransactionFrequency: float
    recurringTransactionCount: int
    averageTransactionFrequencyPerDay: float
    transactionVelocityPattern: float

class CreditScoringModelInput(BaseModel):
    userProfile: UserProfileData
    transactionMetrics: TransactionMetricsInput

class ScoreBreakdownItem(BaseModel):
    score: float = Field(..., ge=0, le=100)
    weight: int

class ScoreBreakdownOutput(BaseModel):
    paymentHistory: ScoreBreakdownItem
    creditUtilization: ScoreBreakdownItem
    lengthOfHistory: ScoreBreakdownItem
    newCredit: ScoreBreakdownItem
    creditMix: ScoreBreakdownItem
    alternativeFactors: ScoreBreakdownItem

class ImprovementTipOutput(BaseModel):
    category: str
    suggestion: str
    impactLevel: str = Field(..., pattern="^(low|medium|high)$")

class CreditScoringModelOutput(BaseModel):
    creditScore: int = Field(..., ge=300, le=850)
    confidenceScore: float = Field(..., ge=0, le=100)
    riskLevel: str = Field(..., pattern="^(low|medium|high)$")
    riskCategory: str = Field(..., pattern="^(excellent|good|fair|poor|very_poor)$")
    scoreBreakdown: ScoreBreakdownOutput
    recommendations: List[str]
    improvementTips: List[ImprovementTipOutput]

# --- Inference Classes (Copied from training scripts) ---

class AnomalyDetectionInference:
    def __init__(self, model, scaler, encoders, feature_columns, iso_forest):
        self.model = model
        self.scaler = scaler
        self.le_category = encoders['category']
        self.le_city = encoders['city']
        self.le_merchant = encoders['merchant']
        self.feature_columns = feature_columns
        self.iso_forest = iso_forest
        
    def preprocess_transaction(self, transaction_data: TransactionDataInput, contextual_features: ContextualFeaturesInput, user_behavioral_context: UserBehavioralContextInput):
        input_data = {
            'transactionId': transaction_data.transactionId,
            'amount': transaction_data.amount,
            'type': transaction_data.type,
            'category': transaction_data.category,
            'subcategory': transaction_data.subcategory,
            'hourOfDay': transaction_data.hourOfDay,
            'dayOfWeek': transaction_data.dayOfWeek,
            'isWeekend': transaction_data.isWeekend,
            'isLateNight': transaction_data.isLateNight,
            'merchant': transaction_data.merchant,
            'latitude': transaction_data.location.latitude,
            'longitude': transaction_data.location.longitude,
            'city': transaction_data.location.city,
            'state': transaction_data.location.state,
            'paymentMethod': transaction_data.paymentMethod,
            'balanceAfterTransaction': transaction_data.balanceAfterTransaction,
            
            'timeSinceLastTransaction': contextual_features.timeSinceLastTransaction,
            'avgAmountForCategoryLast7Days': contextual_features.avgAmountForCategoryLast7Days,
            'stdDevAmountForCategoryLast30Days': contextual_features.stdDevAmountForCategoryLast30Days,
            'userAvgTransactionAmount': contextual_features.userAvgTransactionAmount,
            'userStdDevTransactionAmount': contextual_features.userStdDevTransactionAmount,
            'locationDeviationFromUsualPatterns': contextual_features.locationDeviationFromUsualPatterns,
            'merchantFrequencyForUser': contextual_features.merchantFrequencyForUser,
            'consecutiveTransactionsSameMerchant': contextual_features.consecutiveTransactionsSameMerchant,
            'velocityOfTransactionsInShortPeriod': contextual_features.velocityOfTransactionsInShortPeriod,
            'unusualTimePattern': int(contextual_features.unusualTimePattern),
            'geographicalOutlier': int(contextual_features.geographicalOutlier),
            'amountOutlierForCategory': int(contextual_features.amountOutlierForCategory),
            'merchantOutlier': int(contextual_features.merchantOutlier),
            
            'avgMonthlySpending': user_behavioral_context.avgMonthlySpending,
            'typicalTransactionCountPerDay': user_behavioral_context.typicalTransactionCountPerDay,
            'historicalAnomalyRateForUser': user_behavioral_context.historicalAnomalyRateForUser,
            'averageTransactionAmountLast30Days': user_behavioral_context.averageTransactionAmountLast30Days
        }
        
        input_data['amountDeviationFromUserAvg'] = abs(input_data['amount'] - input_data['userAvgTransactionAmount'])
        
        try:
            input_data['category_encoded'] = self.le_category.transform([input_data['category']])[0]
        except ValueError:
            input_data['category_encoded'] = 0
            
        try:
            input_data['city_encoded'] = self.le_city.transform([input_data['city']])[0]
        except ValueError:
            input_data['city_encoded'] = 0
            
        try:
            input_data['merchant_encoded'] = self.le_merchant.transform([input_data['merchant']])[0]
        except ValueError:
            input_data['merchant_encoded'] = 0
        
        feature_vector = []
        for feature in self.feature_columns:
            feature_vector.append(input_data.get(feature, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        return feature_vector_scaled
    
    def predict(self, transaction_data: TransactionDataInput, contextual_features: ContextualFeaturesInput, user_behavioral_context: UserBehavioralContextInput) -> AnomalyDetectionModelOutput:
        X_processed = self.preprocess_transaction(transaction_data, contextual_features, user_behavioral_context)
        
        predictions = self.model.predict(X_processed, verbose=0)
        reconstruction = predictions[0]
        binary_prob = float(predictions[1][0])
        anomaly_score = float(predictions[2][0] * 100)
        
        reconstruction_error = np.mean(np.square(X_processed - reconstruction))
        
        iso_score = self.iso_forest.decision_function(X_processed)[0]
        iso_normalized = 1 - (iso_score - (-0.5)) / (0.5 - (-0.5))
        
        ensemble_score = (binary_prob + iso_normalized) / 2
        
        is_anomaly = ensemble_score > 0.5
        
        if ensemble_score >= 0.8:
            fraud_risk = "critical"
        elif ensemble_score >= 0.6:
            fraud_risk = "high"
        elif ensemble_score >= 0.4:
            fraud_risk = "medium"
        else:
            fraud_risk = "low"
        
        detected_patterns = self.detect_patterns(transaction_data, contextual_features, user_behavioral_context)
        risk_factors = self.calculate_risk_factors(transaction_data, contextual_features, user_behavioral_context, ensemble_score)
        anomaly_reasons = self.generate_anomaly_reasons(transaction_data, contextual_features, user_behavioral_context, ensemble_score)
        
        return AnomalyDetectionModelOutput(
            transactionId=transaction_data.transactionId,
            isAnomaly=bool(is_anomaly),
            anomalyScore=min(100, max(0, anomaly_score)),
            fraudRisk=fraud_risk,
            detectedPatterns=detected_patterns,
            riskFactors=risk_factors,
            anomalyReasons=anomaly_reasons,
            reconstructionError=float(reconstruction_error),
            ensembleScore=float(ensemble_score * 100)
        )
    
    def detect_patterns(self, transaction_data: TransactionDataInput, contextual_features: ContextualFeaturesInput, user_behavioral_context: UserBehavioralContextInput) -> List[str]:
        patterns = []
        
        if transaction_data.amount > user_behavioral_context.averageTransactionAmountLast30Days * 5:
            patterns.append("unusually_high_amount")
        
        if transaction_data.isLateNight:
            patterns.append("late_night_transaction")
        
        if contextual_features.geographicalOutlier:
            patterns.append("geographic_outlier")
        
        if contextual_features.merchantOutlier:
            patterns.append("unfamiliar_merchant")
        
        if contextual_features.velocityOfTransactionsInShortPeriod > 5:
            patterns.append("high_transaction_velocity")
        
        if transaction_data.isWeekend and transaction_data.amount > user_behavioral_context.avgMonthlySpending * 0.1:
            patterns.append("unusual_weekend_spending")
        
        return patterns
    
    def calculate_risk_factors(self, transaction_data: TransactionDataInput, contextual_features: ContextualFeaturesInput, user_behavioral_context: UserBehavioralContextInput, ensemble_score: float) -> List[RiskFactorOutput]:
        risk_factors = []
        
        amount_deviation = abs(transaction_data.amount - user_behavioral_context.averageTransactionAmountLast30Days)
        amount_weight = min(100, (amount_deviation / (user_behavioral_context.averageTransactionAmountLast30Days + 1e-6)) * 50)
        
        if amount_weight > 20:
            risk_factors.append(RiskFactorOutput(
                factor='unusual_amount',
                weight=float(amount_weight),
                description=f"Transaction amount deviates significantly from user's average"
            ))
        
        if transaction_data.isLateNight:
            risk_factors.append(RiskFactorOutput(
                factor='unusual_time',
                weight=30.0,
                description="Transaction occurred during unusual hours"
            ))
        
        if contextual_features.locationDeviationFromUsualPatterns > 100:
            location_weight = min(100, contextual_features.locationDeviationFromUsualPatterns / 10)
            risk_factors.append(RiskFactorOutput(
                factor='unusual_location',
                weight=float(location_weight),
                description=f"Transaction location is far from user's usual area"
            ))
        
        if contextual_features.merchantOutlier:
            risk_factors.append(RiskFactorOutput(
                factor='unfamiliar_merchant',
                weight=25.0,
                description="Transaction with an unfamiliar merchant"
            ))
        
        if contextual_features.velocityOfTransactionsInShortPeriod > 3:
            velocity_weight = min(100, contextual_features.velocityOfTransactionsInShortPeriod * 15)
            risk_factors.append(RiskFactorOutput(
                factor='high_velocity',
                weight=float(velocity_weight),
                description="High number of transactions in short time period"
            ))
        
        return risk_factors
    
    def generate_anomaly_reasons(self, transaction_data: TransactionDataInput, contextual_features: ContextualFeaturesInput, user_behavioral_context: UserBehavioralContextInput, ensemble_score: float) -> List[AnomalyReasonOutput]:
        reasons = []
        
        if ensemble_score > 0.7:
            if transaction_data.amount > user_behavioral_context.averageTransactionAmountLast30Days * 3:
                reasons.append(AnomalyReasonOutput(
                    reason='Transaction amount is significantly higher than usual',
                    severity='high',
                    confidence=90
                ))
            
            if transaction_data.isLateNight:
                reasons.append(AnomalyReasonOutput(
                    reason='Transaction occurred during unusual hours (late night)',
                    severity='medium',
                    confidence=80
                ))
            
            if contextual_features.locationDeviationFromUsualPatterns > 200:
                reasons.append(AnomalyReasonOutput(
                    reason='Transaction location is very far from usual area',
                    severity='high',
                    confidence=85
                ))
        
        elif ensemble_score > 0.5:
            if contextual_features.merchantOutlier:
                reasons.append(AnomalyReasonOutput(
                    reason='Transaction with an unfamiliar merchant',
                    severity='medium',
                    confidence=70
                ))
            
            if transaction_data.isWeekend and transaction_data.amount > user_behavioral_context.avgMonthlySpending * 0.05:
                reasons.append(AnomalyReasonOutput(
                    reason='Unusual spending pattern on weekend',
                    severity='low',
                    confidence=60
                ))
        
        return reasons

class CreditScoringInference:
    def __init__(self, model, scaler, encoders, feature_columns):
        self.model = model
        self.scaler = scaler
        self.le_employment = encoders['employment']
        self.le_account_type = encoders['account_type']
        self.le_risk_category = encoders['risk_category']
        self.le_risk_level = encoders['risk_level']
        self.feature_columns = feature_columns
        
    def preprocess_input(self, user_profile: UserProfileData, transaction_metrics: TransactionMetricsInput):
        input_data = {**user_profile.model_dump(), **transaction_metrics.model_dump()}
        
        input_data['employmentType_encoded'] = self.le_employment.transform([input_data['employmentType']])[0]
        input_data['bankAccounts_accountType_encoded'] = self.le_account_type.transform([input_data['bankAccounts_accountType']])[0]
        
        if 'existingLoans_loanTypes' in input_data:
            input_data['existingLoans_activeLoanCount'] = len(input_data['existingLoans_loanTypes'])
        
        feature_vector = []
        for feature in self.feature_columns:
            feature_vector.append(input_data.get(feature, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        return feature_vector_scaled
    
    def predict(self, user_profile: UserProfileData, transaction_metrics: TransactionMetricsInput) -> CreditScoringModelOutput:
        X_processed = self.preprocess_input(user_profile, transaction_metrics)
        
        predictions = self.model.predict(X_processed, verbose=0)
        
        credit_score = int(predictions[0][0])
        confidence_score = float(predictions[1][0] * 100)
        risk_category_idx = np.argmax(predictions[2][0])
        risk_level_idx = np.argmax(predictions[3][0])
        
        risk_category = self.le_risk_category.inverse_transform([risk_category_idx])[0]
        risk_level = self.le_risk_level.inverse_transform([risk_level_idx])[0]
        
        score_breakdown = ScoreBreakdownOutput(
            paymentHistory=ScoreBreakdownItem(score=float(user_profile.paymentRegularity), weight=35),
            creditUtilization=ScoreBreakdownItem(score=max(0, 100 - user_profile.creditUtilizationRatio * 1.5), weight=30),
            lengthOfHistory=ScoreBreakdownItem(score=min(100, user_profile.bankAccounts_accountAge * 3.33), weight=15),
            newCredit=ScoreBreakdownItem(score=max(0, 100 - user_profile.creditCards_cardCount * 10), weight=10),
            creditMix=ScoreBreakdownItem(score=min(100, user_profile.existingLoans_activeLoanCount * 20), weight=10),
            alternativeFactors=ScoreBreakdownItem(score=(user_profile.digitalPaymentScore + user_profile.socialMediaScore + user_profile.appUsageScore) / 3, weight=10)
        )
        
        recommendations = self.generate_recommendations(user_profile, credit_score)
        improvement_tips = self.generate_improvement_tips(user_profile, score_breakdown)
        
        return CreditScoringModelOutput(
            creditScore=credit_score,
            confidenceScore=confidence_score,
            riskLevel=risk_level,
            riskCategory=risk_category,
            scoreBreakdown=score_breakdown,
            recommendations=recommendations,
            improvementTips=improvement_tips
        )
    
    def generate_recommendations(self, user_profile: UserProfileData, credit_score: int) -> List[str]:
        recommendations = []
        
        if credit_score < 650:
            recommendations.append("Focus on improving payment history by paying all bills on time")
            recommendations.append("Reduce credit card utilization below 30%")
        
        if user_profile.creditUtilizationRatio > 50:
            recommendations.append("Pay down existing credit card balances")
        
        if user_profile.existingLoans_activeLoanCount > 5:
            recommendations.append("Consider consolidating multiple loans")
        
        if user_profile.bankAccounts_accountAge < 2:
            recommendations.append("Maintain older accounts to improve credit history length")
        
        return recommendations
    
    def generate_improvement_tips(self, user_profile: UserProfileData, score_breakdown: ScoreBreakdownOutput) -> List[ImprovementTipOutput]:
        tips = []
        
        if score_breakdown.paymentHistory.score < 70:
            tips.append(ImprovementTipOutput(
                category='Payment History',
                suggestion='Set up automatic payments to ensure all bills are paid on time',
                impactLevel='high'
            ))
        if score_breakdown.creditUtilization.score < 70:
            tips.append(ImprovementTipOutput(
                category='Credit Utilization',
                suggestion='Keep credit card balances below 30% of available credit',
                impactLevel='high'
            ))
        if score_breakdown.lengthOfHistory.score < 70:
            tips.append(ImprovementTipOutput(
                category='Credit History',
                suggestion='Keep old accounts open to maintain credit history length',
                impactLevel='medium'
            ))
        
        return tips

# Initialize inference pipelines
anomaly_inference_pipeline = AnomalyDetectionInference(
    anomaly_model, anomaly_scaler, anomaly_encoders, anomaly_feature_columns, anomaly_iso_forest
)

credit_inference_pipeline = CreditScoringInference(
    credit_model, credit_scaler, credit_encoders, credit_feature_columns
)

# --- FastAPI Endpoints ---

@app.post("/predict-credit-score", response_model=CreditScoringModelOutput, summary="Predict Credit Score")
async def predict_credit_score(input_data: CreditScoringModelInput):
    """
    Calculates a credit score based on user financial profile and aggregated transaction metrics.
    """
    try:
        result = credit_inference_pipeline.predict(input_data.userProfile, input_data.transactionMetrics)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Credit scoring prediction failed: {e}")

@app.post("/predict-anomaly", response_model=AnomalyDetectionModelOutput, summary="Detect Transaction Anomaly")
async def predict_anomaly(input_data: AnomalyDetectionModelInput):
    """
    Identifies suspicious transaction patterns and provides an anomaly score and risk factors.
    """
    try:
        result = anomaly_inference_pipeline.predict(
            input_data.transactionData,
            input_data.contextualFeatures,
            input_data.userBehavioralContext
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection prediction failed: {e}")

@app.get("/health", summary="Health Check")
async def health_check():
    """
    Checks the health of the API and model loading status.
    """
    return {"status": "ok", "message": "ML service is running and models are loaded."}
