# Dataset Information

## Data Structure Overview

The dataset consists of multiple related tables connected via key identifiers. The main application table links to various supplementary tables containing historical credit information.

## Files Description

### application_{train|test}.csv
**Main table** - Static data for all loan applications
- **Train**: Contains TARGET variable (loan default indicator)
- **Test**: Without TARGET (for predictions)
- One row = one loan application
- Contains information about the loan and applicant at application time

### bureau.csv
**Previous credits from other institutions**
- All client's previous credits from other financial institutions reported to Credit Bureau
- Multiple rows per loan (one row per previous credit)
- Linked via: `SK_ID_CURR`

### bureau_balance.csv
**Monthly history of bureau credits**
- Monthly balance snapshots of previous credits in Credit Bureau
- One row per month of history for each previous credit
- Behavioral data over time
- Linked via: `SK_ID_BUREAU`

### previous_application.csv
**Previous Home Credit applications**
- All previous applications for Home Credit loans
- One row per previous application
- Contains previous loan parameters and client info at time of previous application
- Linked via: `SK_ID_CURR`

### POS_CASH_balance.csv
**Monthly POS and cash loan history**
- Monthly balance snapshots of previous POS (point of sales) and cash loans with Home Credit
- One row per month of history for each previous credit
- Behavioral data
- Linked via: `SK_ID_PREV`

### credit_card_balance.csv
**Monthly credit card history**
- Monthly balance snapshots of previous credit cards with Home Credit
- One row per month of history for each previous credit card
- Behavioral data
- Linked via: `SK_ID_PREV`

### installments_payments.csv
**Repayment history**
- Past payment data for previously disbursed Home Credit credits
- One row per payment made OR per missed payment
- One row = one installment payment
- Behavioral data
- Linked via: `SK_ID_PREV`

### HomeCredit_columns_description.csv
**Data dictionary**
- Column descriptions for all tables

## Key Relationships

```
application_{train|test}.csv (SK_ID_CURR)
    ├── bureau.csv (SK_ID_CURR)
    │   └── bureau_balance.csv (SK_ID_BUREAU)
    └── previous_application.csv (SK_ID_CURR)
        ├── POS_CASH_balance.csv (SK_ID_PREV)
        ├── credit_card_balance.csv (SK_ID_PREV)
        └── installments_payments.csv (SK_ID_PREV)
```

## Key Identifiers

- **SK_ID_CURR**: Current application ID (links to main application table)
- **SK_ID_BUREAU**: Bureau credit ID (links bureau to bureau_balance)
- **SK_ID_PREV**: Previous application ID (links previous_application to payment history tables)

## Target Variable

- **TARGET**: Binary variable indicating loan default
  - 1 = Client had payment difficulties
  - 0 = All payments on time
  - Only available in training set
