# (c) Mikal Rian 2025
# 
# To do: Include valuation (growth) of real estate → the price will go up over time, adding x months to n_months (time to save).
# Note: We're rounding down the floats with int(x)
#
import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

N = 480  # months to graph
n = int
#n_months_debt = 24

def graph(g, y_values, x_axis, y_min=0, y_max=31, y_step=1, x_label='days', y_label='reps', style='-', color=None, legend='', loc_legend='upper right'):
	g.set_xlabel(x_label)
	g.set_ylabel(y_label)
	g.plot(x_axis, y_values, style, color=color, label=legend)
	g.set_yticks(np.arange(y_min, y_max, y_step)) #; g.set_xticks(ticks=x_axis)
	g.legend(loc=loc_legend) # ; return g

# Remember: Depending on loan terms, payment recalculations may or may not be allowed (that is, adjusting future monthly payments).
def amort_shcedule(P, yr, n, extra=0):
	if yr == 0:
		M = P/n
		r = 0
	else:
		r = yr/12
		M = (P * r * (1+r)**n) / ((1+r)**n-1)
	interest_payments = []
	principal_payments = []
	Ps = []
	for j in (range(n)):
		ip = P*r
		pp = M-ip+extra
		P = P-pp#-extra
		interest_payments.append(ip)
		principal_payments.append(pp)
		Ps.append(P if P >= 0 else 0)
		if P <= 0:
			break
	return interest_payments, principal_payments, Ps

def get_debt_array(start, end, length, debt, repay):
	N = length
	debt_array = [0 for i in range(N)]
	#start = int(start)
	debt_array[start-1] = -debt
	for i in range(start, N):
		#k = n_months_to_cap+i
		d = debt_array[i-1]
		s = d + repay
		s = 0 if s > 0 else s
		debt_array[i] = s
	debt_array[start-1] = 0
	return debt_array

def get_future_value(start, end, init_value, rate):
	y = [0 for i in range(end)]
	y[start-1] = init_value
	for i in range(start, end):
		y[i] = y[i-1] * rate
	return y

def get_r(yr):
	r = np.e**(np.log((1+yr))/12)
	return r

def geometric_series_plus_more(start, end, N, monthly, yearly_rate):
	y = [0 for i in range(N)]
	n = end-start
	r = np.e**(np.log((1+yearly_rate))/12)
	for i in range(start, end):
		n = i - start + 1 + 1 # We want n to start on 2, because we want the first month with investment to also compound (we invest at 00:01 the first day, and read the result at 23:59 the last day).
		a = monthly * (1-r**n) / (1-r)
		a -= monthly
		y[i] = a
	for i in range(end, N):
		y[i] = y[i-1] * r
	return y

def geometric_series(n, start_value=0, repeating_amount=0, periodic_rate=1):
	r = periodic_rate
	if n == 0:
		return []
	y = [0 for i in range(n)]
	if start_value == 0:
		start_value = repeating_amount
	y[0] = start_value * periodic_rate #+ repeating_amount * periodic_rate
	if r == 1:  # Makes .../(1-r) zero division.
		for i in range(1, n):
			y[i] = start_value * periodic_rate + repeating_amount * i
	else:
		#for i in range(0, n):
		#	m = i + 2
		#	a = repeating_amount * (1-r**m) / (1-r)
		#	a -= repeating_amount
		#	y[i] = a
		for i in range(1, n):
			#y[i] = (y[i-1] + repeating_amount) * periodic_rate
			y[i] = y[i-1] * periodic_rate + repeating_amount
	return y

#k = geometric_series(10, 1, 0, 1.07)
#k = geometric_series(10, 1, 0, 1.07)
#print(k); exit()

def compounding(start_value, n, periodic_rate):
	y = [0 for i in range(n)]
	r = periodic_rate
	y[0] = start_value * periodic_rate
	for i in range(1, n):
		y[i] = y[i-1] * r
	return y

def get_debt(i, P, n, extra=0):
	r = i/12
	if i == 0:
		return P
	else:
		try:
			monthly_installments = P * (r*(1 + r)**n) / ((1 + r)**n - 1)
		except Exception as e:
			#monthly_installments = (capital_house - capital_goal) / n
			raise e
		return monthly_installments * n

buy0, rent, space, display = st.columns([2, 2, 1, 7])

with buy0:
	st.write('Real estate')
	capital_house = st.number_input('Total value', value=300, key='capital_house')
	capital_goal = st.number_input('Capital goal', value=90, key='capital_goal')
	capital_current = st.number_input('Current capital', value=90, key='capital_current')
	capital_to_earn = capital_goal - capital_current
	monthly_savings = st.number_input('Monthly savings', min_value=0.01, value=2.1, key='monthly_savings')

	try:
		n_months = capital_to_earn/monthly_savings
	except Exception as e:
		# "Monthly repayment or savings must be higher than 0!"
		n_months = 999

	n_months_to_cap = n_months

	bank_interest = st.slider('\\% loan interest', min_value=0, max_value=20, value=10, step=1, key="bank_interest")
	bank_interest = bank_interest/100
	percent_growth_real_estate = st.slider('\\% growth asset, post acquisition', min_value=-5, max_value=15, value=1, step=1, key="g_real_estate")
	rate_growth_realestate = percent_growth_real_estate/100 + 1
	rate_growth_realestate = np.e**(np.log(rate_growth_realestate)/12)

	n_months_debt = st.slider('N months to repay', min_value=6, max_value=360, value=240, step=6, key='n_months_repay')

# Debt
Debt = get_debt(i=bank_interest, P=capital_house-capital_goal, n=n_months_debt)
monthly_installments = Debt/n_months_debt

with buy0:
	# good: st.write('Months to cap goal:', round(n_months_to_cap, 2))#, 'Monthly installments:', round(monthly_installments, 2))
	#st.write('Monthly installments:', round(monthly_installments, 2))
	# auto: invest_post_acq = st.number_input('Invest post asset acquisition', value=float(monthly_savings) - monthly_installments, key='invest_post_acq')

	#try:
	#	repay_amount = st.number_input('Monthly installments', value=monthly_installments, min_value=monthly_installments, max_value=monthly_savings, key='repay_amount', disabled=True)
	#except Exception as e:
	#	st.write("Monthly installments: ", round(monthly_installments, 2))
	#	repay_amount = monthly_installments
	#diff = monthly_savings-repay_amount

	diff = monthly_savings - monthly_installments
	invest_post_acq = diff if diff >= 0 else 0
	st.write("Monthly installments: ", round(monthly_installments, 2))
	#if diff <= 0:
	#	invest_post_acq = st.number_input('Invest post asset acquisition', value=0, key='invest_post_acq', disabled=True)
	#else:
	#	invest_post_acq = st.number_input('Invest post asset acquisition', min_value=diff, value=diff, key='invest_post_acq')
	st.write('Invest or extra repay:', round(invest_post_acq,2))

with rent:
	st.write('Rent & invest')
	monthly_invest0 = st.number_input(f'Invest x {int(n_months)} months (cap goal)', value=monthly_savings, key='monthly_invest0', disabled=True)
	## monthly_rent = st.number_input('Monthly rent (post cap goal)', value=0., key='montly_rent', disabled=True)
	# auto monthly_invest1 = st.number_input('Monthly invest (post cap goal)', value=monthly_invest0-monthly_rent, key='montly_invest1')
	imax = invest_post_acq + monthly_installments
	# auto: monthly_invest1 = st.number_input('Monthly invest (post cap goal)', max_value=imax, value=float(invest_post_acq), key='montly_invest1')
	## monthly_invest1 = st.number_input('Monthly invest (post cap goal)', value=.60, key='montly_invest1')
	invest_portion = st.slider('Invest \\% of monthly savings', min_value=0, max_value=100, value=33, key='invest_portion')
	monthly_invest1 = monthly_savings * (invest_portion/100)
	#st.write("monthly rent: ", monthly_rent)
	#montly_invest1 = st.text_input('Monthly invest', value=0, key='rent')


# Calculate capital for end of each month (suppose saving starts on the 0th second at the beginning).
n = int(n_months_to_cap)  # Don't like using int() like this here...

n_total_duration_of_activity = int(n_months_to_cap + n_months_debt)
st.write('Total activity duration', n_total_duration_of_activity)

# Real estate
#payments_interest, payments_principal, principals = amort_shcedule(P=capital_house - capital_goal, yr=bank_interest, n=n_months, extra=repay_amount)
start = int(n_months_to_cap) # I don't like using int() like this...
#extra = invest_post_acq # repay_amount - monthly_installments

payments_interest, payments_principal, principals = amort_shcedule(P=capital_house - capital_goal, yr=bank_interest, n=int(n_months_debt), extra=invest_post_acq)
n_months_duration_debt0 = len(principals)
#print("n_months_duration_debt0", n_months_duration_debt0)
repay_total = payments_interest[0] + payments_principal[0]
debt_array_with_extra_pay = get_debt_array(start=start, end=n_months_duration_debt0, length=N, debt=sum(payments_interest) + sum(payments_principal), repay=repay_total)
#print(debt_array_with_extra_pay); print(len(debt_array_with_extra_pay)); exit()
total_cost_fast_repay = sum(payments_principal) + sum(payments_interest)

payments_interest, payments_principal, principals = amort_shcedule(P=capital_house - capital_goal, yr=bank_interest, n=int(n_months_debt), extra=0)
n_months_duration_debt1 = len(principals)  # Must be the same as n_months_debt
#print("debt1 ", n_months_duration_debt1)
debt_array_regular = get_debt_array(start=start, end=n_months_duration_debt1, length=N, debt=sum(payments_interest) + sum(payments_principal), repay=monthly_installments)
total_cost_regular = sum(payments_principal) + sum(payments_interest)


with rent:
	percent_growth = st.slider('\\% growth', min_value=0, max_value=15, value=4, step=1, key='g_invest')
	rate_growth = percent_growth/100 + 1
	rate_growth = np.e**(np.log(rate_growth)/12)
	st.write("Amount invested:", round(monthly_invest1,2))
	st.divider()
	st.divider()
	st.divider()
	#display_N = st.slider('Display months:', min_value=60, max_value=480, value=300, step=60, key='display_N')
	display_N = int(n_months_to_cap) + st.selectbox('Display months', [i for i in range(120, 481, 120)], index=1)
	display_N = display_N if display_N <= N else N


with buy0:
	percent_growth_real_invest = st.slider('\\% growth other investment', min_value=0, max_value=15, value=percent_growth, step=1, key='g_real_and_invest', disabled=True)


future_value = get_future_value(start=int(n_months_to_cap), end=N, init_value=capital_house, rate=rate_growth_realestate)
#print('futu', future_value); exit()

net_worth_real_and_extra = [a + b for a, b in zip(future_value, debt_array_with_extra_pay)]
#print("net", net_worth_real_and_extra)
net_worth_real_and_extra[0:int(n_months_to_cap)] = [monthly_savings * (i+1) for i in range(int(n_months_to_cap))]
# So far, so good. 
#print('net', net_worth_real_and_extra); exit()
#with display:
#	st.write(net_worth_real_and_extra)
stage00 = [0 for i in range(int(n_months_to_cap) + n_months_duration_debt0)]
# good
# From debt repayment date, we invest the whole amount.
#stage01 = geometric_series(n = n_months_debt - n_months_duration_debt0 - int(n_months_to_cap), repeating_amount = monthly_savings, periodic_rate = rate_growth)
#duration = n_total_duration_of_activity - n_months_duration_debt0
duration = n_total_duration_of_activity - len(stage00) #- n_months_duration_debt0
#print('duration geo s', duration)
#print('len(stage00)', len(stage00))
#print("stage00", stage00)
#print("n_months_duration_debt0: ", n_months_duration_debt0)
stage01 = geometric_series(n=duration, repeating_amount=monthly_savings, periodic_rate=rate_growth) # Maybe duration should be from n_months_debt1 instead? 
#print('len(stage01)', len(stage01))#; exit()
#print(stage01)#; exit()
#stage02 = compounding(start_value = stage01[-1], n= N - len(stage01), periodic_rate = rate_growth)
#print("n_total_duration_of_activity", n_total_duration_of_activity)
#print("N - n_total_duration_of_activity", N - n_total_duration_of_activity)
if len(stage01) == 0:  # If monthly_savings - monthly_installments == 0 or less, then start_value = 0.
	start_value = 0
else:
	start_value = stage01[-1]
#print('start value = ', start_value)
stage02 = compounding(start_value = start_value, n = N, periodic_rate = rate_growth)  # len(stage01) should be same as N - n_total_duration_of_activity
#print('stage02', stage02); exit()
stages = stage00
stages.extend(stage01)
#print(stages); exit()
stages.extend(stage02)
#print('stages', stages); print()#; exit()
#print(net_worth_real_and_extra)
net_worth_real_and_extra = [a + b for a, b in zip(net_worth_real_and_extra, stages)]
#print("length: ", len(net_worth_real_and_extra)); exit()
#print()
#print(net_worth_real_and_extra)
# So far so good? 

#print("length: ", len(net_worth_real_and_extra)); exit()

#print(net_worth_real_and_extra); exit()
#print("debt ", debt_array_with_extra_pay[0:24])
#print("futu ", future_value[0:24]); 
#print("net  ", net_worth_real_and_extra[0:24]); exit()
#with display:
#	st.write('stages', stages)
#with display:
#	st.write("net", net_worth_real_and_extra)


# Now let's add extra investments (if afforded)
# But it's going too far — until +n_months_to_cap (So 240 months with debt repay (invest) becomes 240+n_months_to_cap)
net_worth_real_and_invest = [a + b for a, b in zip(future_value, debt_array_regular)]
#with display:
#	st.write(net_worth_real_and_invest)
net_worth_real_and_invest[0:int(n_months_to_cap)] = [monthly_savings * (i+1) for i in range(int(n_months_to_cap))] 
#with display:
#	st.write(net_worth_real_and_invest)
stage01 = geometric_series(n = n_months_debt - int(n_months_to_cap), repeating_amount = invest_post_acq, periodic_rate = rate_growth)
stage02 = compounding(start_value = stage01[-1], n= N - len(stage01), periodic_rate = rate_growth)
stages = stage01
stages.extend(stage02)
net_worth_real_and_invest = [a + b for a, b in zip(net_worth_real_and_invest, stages)]

#with display:
#	st.write('stages', stages)
#investments1 = geometric_series(start=3, end=15, monthly=1, yearly_rate=.07)
#print("net", net_worth_real_and_invest)
#exit()
#net_worth_real_and_invest[0:int(n_months_to_cap)] = [monthly_savings * (i+1) for i in range(int(n_months_to_cap))]
#print("net", net_worth_real_and_invest)


# Invest + rent for n months like in fast repay.
#I_stage01 = geometric_series(n = n_duration_saving, repeating_amount=1, periodic_rate=rate_growth)
I_stage01 = geometric_series(n = n_months_duration_debt0, repeating_amount=monthly_invest1, periodic_rate=rate_growth)
I_stage02 = compounding(start_value=I_stage01[-1], n=N - len(I_stage01), periodic_rate=rate_growth)
Invest = I_stage01
Invest.extend(I_stage02)

# Invest + rent for n months like in n_months_debt
#I_stage01 = geometric_series(n = n_duration_saving, repeating_amount=1, periodic_rate=rate_growth)
#duration = n_months_debt
duration = n_total_duration_of_activity
I_stage00 = geometric_series(n = int(n_months_to_cap), repeating_amount=monthly_savings, periodic_rate=rate_growth)
start_value = I_stage00[-1] if len(I_stage00) > 0 else 0
I_stage01 = geometric_series(n = duration, start_value=start_value, repeating_amount=monthly_invest1, periodic_rate=rate_growth)
#print()
#print("monthly_savings", monthly_savings)
#print(I_stage00[00:50]); exit()
#I_stage01 = [a + b for a, b in zip(I_stage00, I_stage01)]
start_value = I_stage01[-1]# + I_stage01[-1]
print("start_value ", start_value)
print(I_stage00[0:]); print()
print("stage 01", I_stage01); print()
I_stage02 = compounding(start_value=start_value, n=N - len(I_stage01), periodic_rate=rate_growth)
print("stage 02 ", I_stage02[0:])
Invest1 = I_stage00
Invest1.extend(I_stage01)
print()
print(Invest1[0:44])
Invest1.extend(I_stage02)

print(len(net_worth_real_and_extra))
print(len(debt_array_with_extra_pay))
print(len(net_worth_real_and_invest))
print(len(debt_array_regular))
print(len(Invest))



with display:
	#fig, (g1, g2) = plt.subplots(2, 1, figsize=(4, 8), layout='constrained', sharey=False)
	#graph(g1, Invest1, range(len(Invest1)), y_min=1, y_max=300, legend='100kg: Max # reps (best set)', loc_legend='upper left', style='.', color='blue')
	#plt.show()

	avg = np.arange(len(Invest1)).astype('float32')
	avg[:] = 0

	#plt.figure(figsize=(7, 6))		#, dpi=300)
	#plt.plot(Invest1, '-', color='darkgrey')
	#plt.plot(net_worth_real_and_extra, '-', color='blue')
	#plt.plot(debt_array_with_extra_pay, ':', color='red')
	y_min = 0.5
	#plt.yticks(np.arange(y_min, 1.01, 10))
	#st.pyplot(plt.gcf())









	disp_start = 0
	multiplier = int(int(n_months_to_cap) / 12)
	print(multiplier)
	disp_init = 240 + 12*multiplier
	disp_init = disp_init if disp_init <= N else 240
	#print(disp_init); exit()
	#display_N = int(n_months_to_cap) + st.selectbox('Display months', [i for i in range(120, 481, 120)], index=1)
	#disp_start, display_N = st.select_slider('Display months', [i for i in range(0, N+1, 12)], value=(0, disp_init))
	#e = int(n_months_to_cap)
	#display_N += e if (e+display_N) <= N else 0

	data = pd.DataFrame({
	    'x': range(disp_start, display_N),
	    'a: Real (+extra pay, invest) *': net_worth_real_and_extra[disp_start:display_N],
	    'c: Real (+invest) **': net_worth_real_and_invest[disp_start:display_N],
	    'b: debt *': debt_array_with_extra_pay[disp_start:display_N],
	    'd: debt **': debt_array_regular[disp_start:display_N],
	    #f'e: Rent + invest {n_months_duration_debt0}': Invest[disp_start:display_N],
	    f'f: Rent + invest {n_total_duration_of_activity}': Invest1[disp_start:display_N]
	})
	data.set_index('x', inplace=True)
	st.subheader("Real estate & fast repay &nbsp;|&nbsp; Real estate & invest &nbsp;|&nbsp; Invest & rent")
	if (monthly_savings >= monthly_installments):
		st.line_chart(data, color=["#000099", "#ff9999", "#0099ff", "#ffcccc", "#AAAAAA"])
		#st.line_chart(data, color=["#000099", "#ff9999", "#ffcccc", "#AAAAAA"])
		st.write("With extra repayments, your loan repayment will take", n_months_duration_debt0, "months, and total cost will be ", round(total_cost_fast_repay,2))
		st.write("With normal repayments, the loan repayment will take", n_months_debt, "months, and total cost will be ", round(total_cost_regular,2))
	else:
		st.write("Your monthly installments are higher than your saving capability!")
		st.write("Interest rate must be lower, initial capital higher, or N months to repay higher.")
