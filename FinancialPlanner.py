import streamlit as st
from numba import njit, prange
import numpy as np
import pandas as pd


class FinancialPlanner():
    def __init__(self):
        # List of all U.S. states
        self.states = [
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
            "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
            "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
            "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
            "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
            "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
            "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
            "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
            "Washington", "West Virginia", "Wisconsin", "Wyoming"
        ]
        self.income_profile = None
        self.tax_profile = None
        self.after_tax_income_profile = None
        self.real_income_profile = None
        self.risk_profile = None
        self.iterations = 200
    
        self.page_bg_img = "https://consumerfed.org/wp-content/uploads/2020/07/stock-7-8.jpg"
        self.meme_img = "https://lapasseduvent.com/wp-content/uploads/2022/08/meme-stonks.jpg"


# region Page Layout
    def main(self):
        st.set_page_config(page_title="RetirmentModel")
        self.set_custom_background(self.page_bg_img)
        st.title("Financial Planner")
        # github link
        st.markdown('[Check out the github](https://github.com/GrantKincaid/Financial-Planner)', unsafe_allow_html=True)
        st.image(image=self.meme_img)

        self.page_your_information()

        self.page_define_your_investment_profile()

        self.page_expectations_for_future_returns()

        self.page_simulated_portfolio_expectations()

        # Memory collection
        # This is a quick solution to a memory leak issue significalty reduced memory ussage and stoped growth
        del self.__dict__
        print("self", self.__dict__)

    def page_your_information(self):
        mode_options = ["Normal", "Advanced"]
            
        self.mode = st.selectbox("Mode Selection",options=mode_options, index=0)
        if self.mode == "Advanced":
            self.iterations = st.number_input("Number of simulated outcomes",
                                            value=200,
                                            step=100,
                                            min_value=100,
                                            max_value=10_000
                                            )
        st.header("Your Information")
        st.text("All values are Annual")
        st.text(f"All Simulations are run {self.iterations} times")
        # Text Box for monthly income
        self.monthly_income = st.number_input("Gross Yearly Income", 
                                        min_value=0, 
                                        step=1_000, 
                                        value=45_000
                                        ) / 12
        
        self.initial_investment = st.number_input("Initial/Current Investment",
                                                  min_value=0,
                                                  step=1_000,
                                                  value=0,
                                                  )
        
        # Text box for Expected income growth rate
        self.annual_income_growth = st.number_input("Income Growth Rate %",
                                                    min_value=-100.0,
                                                    max_value=1_000.0,
                                                    step=1.0,
                                                    value=5.0,
                                                    ) / 100

        # Create a dropdown (select box) for states
        self.selected_state = st.selectbox("State of Residence:", self.states)
        
        # Text box for Expected working life
        self.length_working_life = st.number_input("Years Working",
                                                   min_value=2,
                                                   max_value=60,
                                                   step=1,
                                                   value=30,
                                                   )
        
        if self.mode == "Advanced":
            self.growth_varinance = st.number_input("Expected Income deviation %",
                                                    min_value=1.0,
                                                    step=0.1,
                                                    value=3.0,
                                                    format="%.1f"
                                                    ) / 100
            
            self.target_income_percentile = st.number_input("Percentile outcome to select",
                                                            min_value=0.0,
                                                            max_value=100.0,
                                                            step=1.0,
                                                            value=50.0
                                                            )
            
            self.inflation_rate = st.number_input("Inflation Rate (Historic 3.16%)",
                                value=3.16,
                                step=0.1
                                ) /100
            
            self.inflation_deviation = st.number_input("Percentage Deviation of Inflation (Historic 3.91%)",
                                                value=3.91,
                                                step=0.1
                                                ) /100
    
        else:
            self.growth_varinance = 1.0 / 100
            self.target_income_percentile = 50
            self.inflation_deviation = 3.91 /100
            self.inflation_rate = 3.16 / 100

        simulated_profiles = self.generate_income_profiles(self.monthly_income, 
                                                            self.annual_income_growth, 
                                                            self.length_working_life,
                                                            variance=self.growth_varinance,
                                                            iterations=self.iterations,
                                                            )
        income_profile = np.percentile(simulated_profiles, self.target_income_percentile, axis=0)
        self.income_profile = income_profile[:, np.newaxis]
        
        self.tax_profile, self.after_tax_income_profile = self.generate_taxed_profiles(self.income_profile, self.selected_state)

        real_income_profiles = self.generate_real_income_profiles(self.after_tax_income_profile.flatten(), self.inflation_rate, self.inflation_deviation, self.iterations)
        self.real_income_profile = np.percentile(real_income_profiles, self.target_income_percentile, axis=0)
        self.real_income_profile = self.real_income_profile[:, np.newaxis]
        
        st.text("Income Profile")
        annulized_income = self.annualize_arr_2D(np.concatenate((self.income_profile, self.tax_profile, self.after_tax_income_profile, self.real_income_profile), axis=1), self.length_working_life)
        joined_profiles = pd.DataFrame(annulized_income * 12)
        joined_profiles = joined_profiles.rename(
            columns={0:"Gross Income", 1:"Taxes Paid", 2:"Income After Taxes", 3:"Real Income"}
            )
        st.line_chart(joined_profiles, x_label="Years", y_label="USD", height=500)


    def page_define_your_investment_profile(self):
        st.header("Define Your Investment Profile")

        # Percentage of income invested
        self.investment_rate = st.number_input("Income to Invest %",
                                               min_value=0.0,
                                               max_value=99.0,
                                               step=1.0,
                                               value=15.0
                                               ) / 100
        
        target_cash = st.number_input("Cash Approximate \%",
                                      value = 1.0,
                                      min_value = 0.5,
                                      step = 1.0
                                      ) / 100
        
        if self.mode == "Advanced":
            st.text("These inputs are to define the way your portfolio will adjust over time")
            starting_risk = st.number_input("Initial Stocks",
                                            value=90.0,
                                            max_value=100.0,
                                            min_value=1.0,
                                            step=0.5
                                            ) / 100
            min_risk = st.number_input("Minimum Stocks",
                                    value=10.0,
                                    max_value=100.0,
                                    min_value=0.0,
                                    step=0.5
                                    )/100
            risk_decay = st.number_input("Stock decay",
                                        value=7.0,
                                        step=0.5
                                        )
            decay_shift = st.number_input("Time Shift",
                                        value=10,
                                        step=1
                                        ) * 12
        
            self.risk_profile = self.sigmoid_decay(starting_risk, self.length_working_life*12, risk_decay, min_risk, decay_shift)
            st.text("Sigmoidal Risk Profile of Investments by Month")
            st.line_chart(self.risk_profile, y_label="USD", x_label="Years", height=500)

        else:
            starting_risk = 90.0 / 100
            min_risk = 10.0 / 100
            risk_decay = 5
            decay_shift = 128
            self.risk_profile = self.sigmoid_decay(starting_risk, self.length_working_life*12, risk_decay, min_risk, decay_shift)

        self.porfolio_profile = self.generate_portfolio_profile(target_cash, self.risk_profile, cash_scaling_factor=target_cash)
        st.text("Portfolio Allocation")
        annual_portfolio = self.annualize_arr_2D(self.porfolio_profile, self.length_working_life)
        porfolio_profile = pd.DataFrame(annual_portfolio)
        porfolio_profile = porfolio_profile.rename(columns={0:"Cash", 1:"Stocks", 2:"Bonds"})
        st.line_chart(porfolio_profile, y_label="USD", x_label="Years", height=500)


    def page_expectations_for_future_returns(self):
            if self.mode == "Advanced":
                st.header("Expectations for Future Returns")
                st.text("These values should be your expected geometric returns for each asset class")

                self.stock_yield_annual = st.number_input("Stocks Expected Annual Yield (Historic 9.38%)",
                                                value=9.38,
                                                min_value=1.0,
                                                max_value=100.0,
                                                step=0.01
                                                ) /100
                    
                self.bond_yield_annual = st.number_input("Bonds Excpected Annual Yield (Historic 5.9%)",
                                                value=5.90,
                                                min_value=0.5,
                                                max_value=25.0,
                                                step=0.1
                                                ) /100
                
                self.risk_free_rate = st.number_input("Risk Free Rate or Yield of Cash Equivalents (Historic 4.79%)",
                                                value=4.79,
                                                min_value=0.1,
                                                max_value = 10.0,
                                                step=0.01,
                                                ) /100
                
                self.stock_variance = st.number_input("Expected Stock Deviation % (Historical S&P 500 is 19.15%)",
                                                value=19.15,
                                                min_value=0.01,
                                                max_value=100.0,
                                                step=0.01
                                                ) /100

                self.bond_varince = st.number_input("Expected Bond Deviation % (Historical 8.4%)",
                                            value=8.4,
                                            min_value=0.5,
                                            max_value=100.0,
                                            step=0.1
                                            ) /100
                
                self.risk_free_rate_variance = st.number_input("Expected Risk free Rate Deviation % (Historical is 7.89%)",
                                                value=7.89,
                                                min_value=0.5,
                                                max_value=25.0,
                                                step=0.01
                                                ) /100
            else:
                self.stock_yield_annual = 9.38 / 100
                self.bond_yield_annual = 5.90 / 100
                self.risk_free_rate = 4.79 / 100
                self.stock_variance = 19.15 / 100
                self.bond_varince = 8.4 / 100
                self.risk_free_rate_variance = 7.89 / 100


    def page_simulated_portfolio_expectations(self):
        st.header("Simulated Portfolio Expectations")

        portfolio_array = self.simulate_portfolio(
            self.after_tax_income_profile, self.porfolio_profile, self.investment_rate,
            self.stock_yield_annual, self.bond_yield_annual, self.risk_free_rate,
            self.stock_variance, self.bond_varince, self.risk_free_rate_variance,
            self.iterations, self.initial_investment
            )

        self.equity_pct = np.percentile(portfolio_array, 50.0, axis=0)
        self.equity_pct = self.annualize_arr_2D(self.equity_pct, self.length_working_life)
        self.equity_pct = pd.DataFrame(self.equity_pct)
        self.equity_pct = self.equity_pct.rename(columns={0:"Stocks", 1:"Bonds", 2:"Cash"})
        st.text("Median Portfolio Make Up")
        st.line_chart(self.equity_pct, y_label="USD", x_label="Years", height=500)

        self.cash_invested = self.after_tax_income_profile * self.investment_rate
        self.cum_cash = np.zeros((self.cash_invested.shape[0], 1), dtype=np.float64)
        for i in range(self.cash_invested.shape[0]):
            self.cum_cash[i] = self.cash_invested[i] + self.cum_cash[i-1]

        self.equity_high = np.sum(np.percentile(portfolio_array, 90.0, axis=0), axis=1)
        self.equity_mid = np.sum(np.percentile(portfolio_array, 50.0, axis=0), axis=1)
        self.equity_low = np.sum(np.percentile(portfolio_array, 10.0, axis=0), axis=1)
        self.equity_spread = np.concatenate((self.equity_high[:, np.newaxis], self.equity_mid[:, np.newaxis], self.equity_low[:, np.newaxis], self.cum_cash), axis=1)
        self.Aequity_spread = self.annualize_arr_2D(self.equity_spread, self.length_working_life)
        self.dfequity_spread = pd.DataFrame(self.Aequity_spread)
        self.dfequity_spread = self.dfequity_spread.rename(columns={0:"90th", 1:"50th", 2:"10th", 3:"Just Cash"})

        st.text("Scenario Outcome Analysis")
        st.line_chart(self.dfequity_spread, y_label="USD", x_label="Years", height=500)


        st.text("Scenario Analysis (Inflation-Adjusted)")
        li_values = [None, None, None, None]
        for i in range(self.equity_spread.shape[1]):
            values = self.generate_real_income_profiles(self.equity_spread[:, i], self.inflation_rate, self.inflation_deviation, self.iterations)
            values = np.percentile(values, self.target_income_percentile, axis=0)
            values = values[:, np.newaxis]
            li_values[i] = values

        self.inflation_adjusted = np.concatenate((li_values[0], li_values[1], li_values[2], li_values[3]), axis=1)
        self.inflation_adjusted = self.annualize_arr_2D(self.inflation_adjusted, self.length_working_life)
        self.pd_inflation_adjusted = pd.DataFrame(self.inflation_adjusted)
        self.pd_inflation_adjusted = self.pd_inflation_adjusted.rename(columns={0:"90th", 1:"50th", 2:"10th", 3:"Just Cash"})
        st.line_chart(self.pd_inflation_adjusted, y_label="USD", x_label="Years", height=500)

    def set_custom_background(self, image_url):
        center_bar_color = "#0e1117"

        st.markdown(
            f"""
            <style>
            /* Background image for the entire app */
            .stApp {{
                background: url("{image_url}") no-repeat center center fixed;
                background-size: cover;
                position: relative;
            }}

            /* Center bar with dynamic width */
            .center-bar {{
                position: fixed;
                top: 0;
                left: 50%;
                transform: translateX(-50%);
                width: clamp(60vw, 70vw, 1100px); /* Adapts width dynamically */
                min-width: 400px;
                max-width: 1100px;
                height: 100%;
                background-color: {center_bar_color};
                z-index: 0;
            }}

            /* Ensure content is above center bar */
            .stApp > div {{
                position: relative;
                z-index: 1;
            }}


            /* Adjust width for smaller screens */
            @media (max-width: 900px) {{
                .center-bar {{
                    width: clamp(60vw, 75vw, 800px);
                }}
            }}

            @media (max-width: 600px) {{
                .center-bar {{
                    width: 90vw;
                }}
            }}

            @media (max-width: 400px) {{
                .center-bar {{
                    width: 100vw;
                }}
            }}
            
            </style>
            <div class="center-bar"></div>
            """,
            unsafe_allow_html=True
        )
# endregion


    @staticmethod
    def generate_portfolio_profile(base_cash_target, risk_profile, cash_scaling_factor=0.1):
        """
        Generate a dynamic portfolio profile based on cash target and risk profile.
        
        Args:
            base_cash_target (float): Base cash target percentage (e.g., 0.2 for 20%).
            risk_profile (np.ndarray): 1D array of risk profile values (0 to 1).
            cash_scaling_factor (float): Factor by which the cash target increases as risk decreases (default=0.1).
        
        Returns:
            np.ndarray: Portfolio profile with columns [cash, risk_assets, safe_assets].
        """
        steps = risk_profile.shape[0]
        portfolio_profile = np.zeros((steps, 3), dtype=np.float64)

        for i in range(steps):
            # Dynamically adjust cash target based on risk profile
            dynamic_cash_target = base_cash_target + cash_scaling_factor * (1 - risk_profile[i])

            # Ensure cash target does not exceed 1
            dynamic_cash_target = min(dynamic_cash_target, 1)

            # Compute portfolio components before normalization
            cash = dynamic_cash_target
            risk_assets = max(risk_profile[i] - cash, 0)
            safe_assets = max((1 - risk_profile[i]) - cash, 0)

            # Normalize the components to ensure they sum to 1
            total = cash + risk_assets + safe_assets
            portfolio_profile[i, 0] = cash / total
            portfolio_profile[i, 1] = risk_assets / total
            portfolio_profile[i, 2] = safe_assets / total

        return portfolio_profile


    @staticmethod
    @njit(parallel=True)
    def simulate_portfolio(
        income_profile, portfolio_ratios, investment_ratio,
        stock_yield, bond_yield, RFR,
        stock_variance, bond_variance, RFR_variance,
        iterations, initial_investment ):
        """
        Cash Flow based simulation of portfolio

        Args:
            income_profile (np.ndarray): 1D array of income over periods n
            portfolio_ratios (np.ndarray) : 2D array of ratios of stocks, bonds, cash
            invetment_ratio (Float): Decimal percentage of income to invest per n
            stock_yield (Float): Decumal percentage of expected stock returns
            bond_yield (Float): Decimal percentage of expected bond returns
            RFR (Float): Decimal percentage of risk free rate
            stock_variance (Float): Decimal standard percentage deviation of stock returns
            bond_variance (Float): Deciaml standard percentage deviation of bond returns
            RFR_variance (Float): Decimal standard percentage deviation of risk free rate returns
            iterations (Int): Number of simulation passes to run
        Returns:
            np.ndarray: results of all simulations struct=[interation, step, [stocks, bonds, cash]]
        """
        steps = income_profile.shape[0]

        # Monthly yields and variances
        n = 12
        m_stock_yield = ((1 + stock_yield)**(1/n))-1 # converting to monthly percentage rate form EAR
        m_bond_yield = ((1 + bond_yield)**(1/n))-1
        m_RFR_yield = ((1 + RFR)**(1/n))-1


        m_std_stock = np.sqrt(stock_variance) / np.sqrt(12)
        m_std_bond = np.sqrt(bond_variance) / np.sqrt(12)
        m_std_RFR = np.sqrt(RFR_variance) / np.sqrt(12)

        # Initialize simulated portfolios (iterations x steps x [stocks, bonds, cash])
        simulated_portfolios = np.zeros((iterations, steps, 3), dtype=np.float64)

        for j in prange(iterations):
            for i in range(steps):
                
                if i > 0:
                    # Stock returns with random fluctuations
                    stock_value = simulated_portfolios[j, i - 1, 0]
                    stock_fluctuations = np.random.normal(0, m_std_stock)
                    simulated_portfolios[j, i - 1, 0] += stock_value * (m_stock_yield + stock_fluctuations)

                    # Bond returns with random fluctuations
                    bond_value = simulated_portfolios[j, i - 1, 1]
                    bond_fluctuations = np.random.normal(0, m_std_bond)
                    simulated_portfolios[j, i - 1, 1] += bond_value * (m_bond_yield + bond_fluctuations)

                    # Cash returns with random fluctuations
                    cash_value = simulated_portfolios[j, i - 1, 2]
                    cash_fluctuations = np.random.normal(0, m_std_RFR)
                    simulated_portfolios[j, i - 1, 2] += cash_value * (m_RFR_yield + cash_fluctuations)

                new_investment_funds = income_profile[i] * investment_ratio
                invested_cash = np.sum(simulated_portfolios[j, i - 1, :]) if i > 0 else 0.0
                
                if i == 0:
                    investment_funds = invested_cash + new_investment_funds + initial_investment
                else:
                    investment_funds = invested_cash + new_investment_funds

                # Add new investment funds based on portfolio ratios
                stock_ratio = portfolio_ratios[i, 1].item()
                bond_ratio = portfolio_ratios[i, 2].item()
                cash_ratio = portfolio_ratios[i, 0].item()

                # Explicit scalar extraction
                simulated_portfolios[j, i, 0] = np.float64(investment_funds[0] * stock_ratio)
                simulated_portfolios[j, i, 1] = np.float64(investment_funds[0] * bond_ratio)
                simulated_portfolios[j, i, 2] = np.float64(investment_funds[0] * cash_ratio)

        return simulated_portfolios


    @staticmethod
    def get_federal_tax(income):
        """Calculate federal tax based on 2023 IRS tax brackets."""
        # 2025 Federal tax brackets (single filer as an example)
        n = 12
        brackets = [
            (0, 0.10, 11925/n),  # 10% on the first $11,925
            (1192/n, 0.12, 48475/n),  # 12% on income over $11,925 up to $48,475
            (48475/n, 0.22, 103350/n),  # 22% on income over $48,475 up to $103,350
            (103350/n, 0.24, 197300/n),  # 24% on income over $103,350 up to $197,300
            (197300/n, 0.32, 250525/n),  # 32% on income over $197,300 up to $250,525
            (250525/n, 0.35, 626350/n),  # 35% on income over $250,525 up to $626,350
            (626350/n, 0.37, float("inf")),  # 37% on income over $626,350
        ]
        
        tax = 0
        for lower, rate, upper in brackets:
            if income > lower:
                taxable = min(income, upper) - lower
                tax += taxable * rate
            if income <= upper:
                break
        return tax
    

    @staticmethod
    def get_state_tax(income, state):
        """Retrieve state tax rates and calculate state tax."""
        # dict of all state tax rates
        state_tax_rates = {
            "Alabama": 0.05,
            "Alaska": 0.00,
            "Arizona": 0.025,
            "Arkansas": 0.049,
            "California": 0.133,
            "Colorado": 0.044,
            "Connecticut": 0.0699,
            "Delaware": 0.066,
            "Florida": 0.00,
            "Georgia": 0.0575,
            "Hawaii": 0.11,
            "Idaho": 0.058,
            "Illinois": 0.0495,
            "Indiana": 0.03,
            "Iowa": 0.038,
            "Kansas": 0.057,
            "Kentucky": 0.045,
            "Louisiana": 0.03,
            "Maine": 0.0715,
            "Maryland": 0.0575,
            "Massachusetts": 0.09,
            "Michigan": 0.0425,
            "Minnesota": 0.0985,
            "Mississippi": 0.044,
            "Missouri": 0.047,
            "Montana": 0.069,
            "Nebraska": 0.052,
            "Nevada": 0.00,
            "New Hampshire": 0.00,
            "New Jersey": 0.1075,
            "New Mexico": 0.059,
            "New York": 0.109,
            "North Carolina": 0.0425,
            "North Dakota": 0.029,
            "Ohio": 0.0399,
            "Oklahoma": 0.05,
            "Oregon": 0.099,
            "Pennsylvania": 0.0307,
            "Rhode Island": 0.0599,
            "South Carolina": 0.07,
            "South Dakota": 0.00,
            "Tennessee": 0.00,
            "Texas": 0.00,
            "Utah": 0.0485,
            "Vermont": 0.0875,
            "Virginia": 0.0575,
            "Washington": 0.00,
            "West Virginia": 0.0482,
            "Wisconsin": 0.0765,
            "Wyoming": 0.00
        }
        
        if state not in state_tax_rates:
            raise ValueError(f"State tax rate not found for {state}.")
        
        state_tax_rate = state_tax_rates[state]
        return income * state_tax_rate
    

    def calculate_total_tax(self, income, state):
        """Calculate total tax (federal + state)."""
        federal_tax = self.get_federal_tax(income)
        state_tax = self.get_state_tax(income, state)
        total_tax = federal_tax + state_tax
        return federal_tax, state_tax, total_tax


    @staticmethod
    @njit
    def generate_income_profiles(
        monthly_income, income_growth, len_working, variance, iterations
    ):
        """
        Generate multiple income profiles using Monte Carlo simulation.
        Returns all simulated profiles for percentile calculation outside the function.
        """
        working_months = len_working * 12
        growth_monthly = ((1 + income_growth)**(1/12))-1 # Convert annual (EAR) growth to monthly percentage rate through compuding this will be tured back into EAR
        std_dev_monthly = np.sqrt(variance) / np.sqrt(12)  # Convert annual variance to monthly std dev

        # Array to store income profiles for all iterations
        simulated_profiles = np.zeros((iterations, working_months), dtype=np.float64)

        # Monte Carlo simulation
        for it in range(iterations):
            income_profile = np.zeros(working_months, dtype=np.float64)
            income_profile[0] = monthly_income
            for i in range(1, working_months):
                # Add random fluctuation based on monthly standard deviation
                random_fluctuation = np.random.normal(0, std_dev_monthly)
                income_profile[i] = income_profile[i - 1] * (1 + growth_monthly + random_fluctuation)
            simulated_profiles[it, :] = income_profile

        return simulated_profiles


    def generate_taxed_profiles(self, monthly_income_profile, state):
        count_months = monthly_income_profile.shape[0]
        tax_profile = np.zeros((count_months, 1), dtype=np.float64)
        after_tax_income_profile = np.zeros((count_months, 1), dtype=np.float64)
        for i in range(0, count_months, 1):
            income = monthly_income_profile[i]
            fed_tax, state_tax, total_tax = self.calculate_total_tax(income=income, state=state)
            tax_profile[i, 0] = total_tax
            after_tax_income_profile[i, 0] = income - total_tax
        return tax_profile, after_tax_income_profile


    @staticmethod
    def sigmoid_decay(initial_value, n, steepness, min_value, shift):
        """
        Generates a sigmoid curve that starts from an initial value and approaches a minimum value over n time steps.
        
        Args:
            initial_value (float): Starting value of the sigmoid.
            n (int): Total number of time steps.
            steepness (float): Controls the steepness of the sigmoid curve.
            min_value (float): The minimum value the sigmoid will approach.
            shift (float): Shifts the sigmoid curve to the right (positive) or left (negative).
        
        Returns:
            np.ndarray: A numpy array of length n containing the sigmoid values.
        """
        # Generate time steps
        time_steps = np.arange(n)
        
        # Calculate normalized sigmoid values (range: 0 to 1)
        normalized_sigmoid = initial_value / (1 + np.exp(steepness * (time_steps - n / 2 - shift) / n))
        
        # Scale to desired range [min_value, initial_value]
        sigmoid_values = min_value + (initial_value - min_value) * normalized_sigmoid
        
        return sigmoid_values


    @staticmethod
    @njit
    def generate_real_income_profiles(income, inflation_rate, inflation_deviation, iterations):
        """
        Generate Monte Carlo simulations of real income profiles adjusted for inflation.
        
        Args:
            income (np.ndarray): 1D array of initial income values (length equal to `length`).
            inflation_rate (float): Average annual inflation rate (as a decimal).
            inflation_deviation (float): Standard deviation of annual inflation rate (as a decimal).
            iterations (int): Number of Monte Carlo iterations.
        
        Returns:
            np.ndarray: 2D array of shape (iterations, length), with each row representing a simulated income profile.
        """
        # Precompute the number of time steps (assuming income is provided for each year)
        time_steps = len(income)

        # Initialize the result array to store income profiles
        real_income_profiles = np.zeros((iterations, time_steps), dtype=np.float64)

        inflation_rate = ((1 + inflation_rate)**(1/12))-1
        inflation_deviation = np.sqrt(inflation_deviation) / np.sqrt(12)


        # Monte Carlo simulation
        for it in range(iterations):
            real_income = np.zeros(time_steps, dtype=np.float64)
            real_income[0] = income[0]
            
            # Start cumulative inflation factor
            inflation_factor = 1.0

            for t in range(1, time_steps):
                # Generate a random inflation rate for this year
                random_inflation = np.random.normal(inflation_rate, inflation_deviation)

                # Update the cumulative inflation factor (compounding effect)
                inflation_factor *= 1 + random_inflation

                # Adjust income based on the cumulative inflation factor
                real_income[t] = income[t] / inflation_factor

            # Store the simulated profile
            real_income_profiles[it, :] = real_income

        return real_income_profiles


# region Tools
    @staticmethod
    def annualize_arr_2D(input_arr, count_years):

        if input_arr.shape[0] % 12 == 0:
            annulized_data = np.zeros((count_years, input_arr.shape[1]), dtype=np.float64)
            for i in range(count_years):
                annulized_data[i, :] = input_arr[i*12, :]

            return annulized_data
        else:
            return input_arr


    @staticmethod
    def annualize_arr_3D(input_arr, count_years):

        if input_arr.shape[0] % 12 == 0: 
            annulized_data = np.zeros((count_years, input_arr.shape[1], input_arr.shape[2]), dtype=np.float64)
            for i in range(count_years):
                annulized_data[i, :, :] = input_arr[i*12, :, :]
            
            return annulized_data
        else:
            return input_arr
# endregion


if __name__ == '__main__':

    fp = FinancialPlanner()
    fp.main()
    