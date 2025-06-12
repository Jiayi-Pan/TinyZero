import os
import json
from datasets import Dataset
from typing import List, Dict
from tqdm import tqdm
import argparse


def load_workfront_data(json_file_path: str) -> List[Dict]:
    samples = []
    
    with open(json_file_path, 'r') as f:
        data_list = json.load(f)
    
    for data in data_list:
        expected_response = data['expected_response']
        
        api_call_steps = format_workfront_api_call(expected_response)
        
        samples.append({
            "question": data['prompt'],
            "answer": api_call_steps,
            "template_type": "workfront_api",
            "original_response": expected_response
        })
    
    return samples


def format_workfront_api_call(expected_response: Dict) -> str:
    return json.dumps(expected_response, indent=2)


def make_prefix(question, template_type='base'):
    """Create the prompt prefix for API tasks"""
    if template_type == 'base':
        prefix = f"""You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront’s custom object names and metadata to do the same using the context given below.

Your role is to interpret a user’s natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy). Your output should follow this structure:

{{
  "prompt": "What are all the tasks with high priority due next week?",
  "expected_response": {{
    "tool": "fetch_data",
    "objCode": "TASK",
    "fields": ["ID", "name", "priority", "plannedCompletionDate"],
    "filters": {{
      "priority": "High",
      "plannedCompletionDate": "$$TODAYbw+7d"
    }}
  }}
}}

 Workfront Object Context You Can Use

 Core Objects:
TASK: Represents individual tasks.
PROJ: Represents projects.
USER: Represents users (people in the org).

--- PROJECT METADATA ---
domain_knowledge:|
  Projects represent the main container of work that is to be completed. A Project may contain many different tasks or issues. Projects have both a plannedCompletionDate 
  which represents an estimate or planned completion date, and also a actualCompletionDate which represents the actual completion date of the project. If you are looking
  for when things actually completed use the actualCompletionDate, if you are wanting to know when something is planned to be completed use the plannedCompletionDate.
  Default Statuses include: Current, Dead, On Hold, Planning, Complete, Requested, Approved, Rejected, and Idea
fields:
  accessorIDs:
    type: array
    description: accessorIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  actualBenefit:
    type: number
    description: Actual Benefit
    verbs: []
  actualBillableExpenseCost:
    type: number
    description: actualBillableExpenseCost
    verbs: []
  actualCompletionDate:
    type: string
    description: The date in which this project was actually completed - this is not the same as the planned completion date which was more of an estimate
    verbs:
      - GET
      - PUT
      - POST
  actualCost:
    type: number
    description: The actual cost of this Project
    verbs:
      - GET
  actualDurationExpression:
    type: string
    description: actualDurationExpression
    verbs: []
  actualDurationMinutes:
    type: integer
    description: Actual Duration Minutes
    verbs: []
  actualExpenseCost:
    type: number
    description: Actual Expense Cost
    verbs: []
  actualHoursLastMonth:
    type: number
    description: Actual Hours Last Month
    verbs: []
  actualHoursLastThreeMonths:
    type: number
    description: Actual Hours Last Three Months
    verbs: []
  actualHoursThisMonth:
    type: number
    description: Actual Hours This Month
    verbs: []
  actualHoursTwoMonthsAgo:
    type: number
    description: Actual Hours Two Months Ago
    verbs: []
  actualLaborCost:
    type: number
    description: Actual Labor Cost
    verbs: []
  actualLaborRevenue:
    type: number
    description: Actual Labor Revenue
    verbs: []
  actualNonBillableExpenseCost:
    type: number
    description: actualNonBillableExpenseCost
    verbs: []
  actualNonLaborCost:
    type: number
    description: Actual Non-Labor Cost
    verbs: []
  actualNonLaborRevenue:
    type: number
    description: Actual Non-Labor Revenue
    verbs: []
  actualRevenue:
    type: number
    description: Actual Revenue
    verbs: []
  actualRiskCost:
    type: number
    description: Actual Risk Cost
    verbs: []
  actualStartDate:
    type: string
    description: The official start date of this project
    verbs:
      - GET
      - PUT
      - POST
  actualValue:
    type: number
    description: Actual Value
    verbs: []
  actualWorkRequired:
    type: integer
    description: Actual Hours
    verbs: []
  actualWorkRequiredDouble:
    type: number
    description: actualWorkRequiredDouble
    verbs: []
  actualWorkRequiredExpression:
    type: string
    description: actualWorkRequiredExpression
    verbs: []
  alignment:
    type: number
    description: Alignment
    verbs: []
  alignmentScoreCardID:
    type: string
    description: Alignment Scorecard ID
    verbs: []
  allApprovedHours:
    type: number
    description: allApprovedHours
    verbs: []
  allUnapprovedHours:
    type: number
    description: allUnapprovedHours
    verbs: []
  approvalCompletionDate:
    type: string
    description: Approval Path Completion Date
    verbs: []
  approvalEstStartDate:
    type: string
    description: Approval Estimated Start Date
    verbs: []
  approvalPlannedStartDate:
    type: string
    description: Approval Planned Start Date
    verbs: []
  approvalPlannedStartDay:
    type: number
    description: approvalPlannedStartDay
    verbs: []
  approvalProcessID:
    type: string
    description: Approval Processes ID
    verbs: []
  approvalProjectedStartDate:
    type: string
    description: Approval Projected Start Date
    verbs: []
  approvalStartDate:
    type: string
    description: Approval Path Start Date
    verbs: []
  approversString:
    type: string
    description: approversString
    verbs: []
  attachedRateCardID:
    type: string
    description: attachedRateCardID
    verbs: []
  attachedRateCardName:
    type: string
    description: attachedRateCardName
    verbs: []
  auditTypes:
    type: array
    description: Audit Types
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ST
          description: Status Change
        - type: string
          enum:
            - AA
          description: Attachment Action
        - type: string
          enum:
            - SC
          description: Scope Change
        - type: string
          enum:
            - "NO"
          description: Note
        - type: string
          enum:
            - GE
          description: General Edit
        - type: string
          enum:
            - CM
          description: Combined Entry
        - type: string
          enum:
            - ER
          description: Error Entry
        - type: string
          enum:
            - SB
          description: Subscription Change
    uniqueItems: true
  autoBaselineRecurOn:
    type: integer
    description: Auto Baseline Recur On
    verbs: []
  autoBaselineRecurrenceType:
    type: string
    description: Auto Baseline Recurrence Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - D
        description: Daily
      - type: string
        enum:
          - W
        description: Weekly
      - type: string
        enum:
          - BW
        description: Bi-weekly
      - type: string
        enum:
          - SM
        description: Semi-monthly
      - type: string
        enum:
          - M
        description: Monthly
      - type: string
        enum:
          - C
        description: Custom
  BCCompletionState:
    type: array
    description: BC Completion State
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - SUMMARY
          description: Summary
        - type: string
          enum:
            - INFO
          description: Project Info
        - type: string
          enum:
            - DEL
          description: Goals
        - type: string
          enum:
            - RISK
          description: Risks
        - type: string
          enum:
            - COST
          description: Costs
        - type: string
          enum:
            - ALIGN
          description: Alignment
        - type: string
          enum:
            - DE
          description: Custom Data
    uniqueItems: true
  bcwp:
    type: number
    description: BCWP
    verbs: []
  bcws:
    type: number
    description: BCWS
    verbs: []
  billedRevenue:
    type: number
    description: Billed Revenue
    verbs: []
  budget:
    type: number
    description: The official budget associated with this Project
    verbs:
      - GET
      - PUT
      - POST
  budgetedCompletionDate:
    type: string
    description: Budgeted Completion Date
    verbs: []
  budgetedCost:
    type: number
    description: Budgeted Cost
    verbs: []
  budgetedHours:
    type: number
    description: Budgeted Hours
    verbs: []
  budgetedLaborCost:
    type: number
    description: Budgeted Labor Cost
    verbs: []
  budgetedLaborCostSource:
    type: string
    description: budgetedLaborCostSource
    verbs: []
  budgetedStartDate:
    type: string
    description: Budgeted Start Date
    verbs: []
  budgetStatus:
    type: integer
    description: Budget Status
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Not Included
      - type: integer
        enum:
          - 1
        description: Included but not Calculated
      - type: integer
        enum:
          - 2
        description: Included and Calculated
  businessCaseStatusLabel:
    type: string
    description: Business Case Status
    verbs: []
  categoryID:
    type: string
    description: Category ID
    verbs: []
  companyID:
    type: string
    description: Company ID
    verbs: []
  companyRatesOverride:
    type: boolean
    description: Company Rates Override
    verbs: []
  completedHours:
    type: number
    description: Completed Hours
    verbs: []
  completedStoryPoints:
    type: number
    description: Completed Story Points
    verbs: []
  completionType:
    type: string
    description: Completion Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - MAN
        description: Manual
      - type: string
        enum:
          - AUT
        description: Automatic
  condition:
    type: string
    description: Condition
    verbs: []
    oneOf:
      - type: string
        enum:
          - "ON"
        description: On Target
      - type: string
        enum:
          - AR
        description: At Risk
      - type: string
        enum:
          - IT
        description: In Trouble
  conditionType:
    type: string
    description: Condition Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - PG
        description: Progress Status
      - type: string
        enum:
          - MN
        description: Manual
  convertedOpTaskEntryDate:
    type: string
    description: Converted Issue Entry Date
    verbs: []
  convertedOpTaskID:
    type: string
    description: convertedOpTaskID
    verbs: []
  convertedOpTaskName:
    type: string
    description: Converted Issue Name
    verbs: []
  convertedOpTaskOriginatorID:
    type: string
    description: convertedOpTaskOriginatorID
    verbs: []
  cpi:
    type: number
    description: Cost Performance Index
    verbs: []
  csi:
    type: number
    description: Cost Schedule Performance Index
    verbs: []
  currency:
    type: string
    description: Currency
    verbs: []
  currentApprovalStepID:
    type: string
    description: Current Approval Stage ID
    verbs: []
  customerID:
    type: string
    description: Customer ID
    verbs: []
  customFormsDescriptions:
    type: string
    description: customFormsDescriptions
    verbs: []
  customFormsGroupIDs:
    type: string
    description: customFormsGroupIDs
    verbs: []
  customFormsIDs:
    type: string
    description: The id of the custom forms attached to this Project
    verbs:
      - GET
  customFormsNames:
    type: string
    description: The name of the custom forms attached to this Project
    verbs:
      - GET
  defaultForbiddenContributeActions:
    type: array
    description: defaultForbiddenContributeActions
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ADD
          description: Add
        - type: string
          enum:
            - EDIT
          description: Edit
        - type: string
          enum:
            - LIMITED_EDIT
          description: Limited Edit
        - type: string
          enum:
            - VIEW
          description: View
        - type: string
          enum:
            - DELETE
          description: Delete
        - type: string
          enum:
            - LEGACY_APPROVE
          description: Approve
        - type: string
          enum:
            - LEGACY_REQUEST_PROJECT
          description: Request
        - type: string
          enum:
            - CHANGE_STATUS
          description: Change Status
        - type: string
          enum:
            - EDIT_CUSTOMDATA
          description: Edit Custom Form
        - type: string
          enum:
            - EDIT_APPROVALPROCESS
          description: Edit Approval Process
        - type: string
          enum:
            - EDIT_ROLE_GROUP
          description: Change a User's Role or Group
        - type: string
          enum:
            - EDIT_ACCESSLEVEL
          description: Change a User's Access Level
        - type: string
          enum:
            - ADD_TASKS
          description: Add Tasks
        - type: string
          enum:
            - ADD_ISSUES
          description: Add Issues
        - type: string
          enum:
            - EDIT_ASSIGNMENTS
          description: Make Assignments
        - type: string
          enum:
            - ADD_DOCUMENTS
          description: Add Documents
        - type: string
          enum:
            - EDIT_FINANCE
          description: Manage Finance
        - type: string
          enum:
            - VIEW_FINANCE
          description: View Finance
        - type: string
          enum:
            - ADD_HOURS
          description: Log Hours
        - type: string
          enum:
            - ADD_EXPENSES
          description: Add Expenses
        - type: string
          enum:
            - SHARE
          description: Share
        - type: string
          enum:
            - PROOF
          description: Proof
        - type: string
          enum:
            - DOWNLOAD
          description: Download
        - type: string
          enum:
            - SHARE_SYSTEMWIDE
          description: Share System-wide
        - type: string
          enum:
            - VIEW_CONTACTINFO
          description: View Contact Info
        - type: string
          enum:
            - ATTACH_CUSTOMFORM
          description: Attach to Custom Data
        - type: string
          enum:
            - BUDGETING_INFORMATION
          description: Edit priorities and budget hours in the Planner
        - type: string
          enum:
            - PLANNED_HOURS_CONTOURING
          description: Update Planned Hours in the Workload Balancer
        - type: string
          enum:
            - ADD_TO_CUSTOM_FORMS
          description: Add to Custom Forms
        - type: string
          enum:
            - ADD_BOOKINGS
          description: Add Bookings
        - type: string
          enum:
            - VIEW_COST_RATES
          description: View Cost Rates
        - type: string
          enum:
            - VIEW_BILLING_RATES
          description: View Billing Rates
        - type: string
          enum:
            - VIEW_GENERAL_FINANCE
          description: View general finance
        - type: string
          enum:
            - EDIT_COST_RATES
          description: Manage Cost Rates
        - type: string
          enum:
            - EDIT_BILLING_RATES
          description: Manage Billing Rates
        - type: string
          enum:
            - EDIT_GENERAL_FINANCE
          description: Manage general finance
        - type: string
          enum:
            - EDIT_TEAMS_I_AM_ON
          description: Edit teams I am on
        - type: string
          enum:
            - EDIT_TEAMS_I_GROUP_ADMIN
          description: Edit teams in groups I manage (Group Admins only)
        - type: string
          enum:
            - EDIT_SYSTEMWIDE
          description: enum.actiontypeenum.editystemwide
    uniqueItems: true
  defaultForbiddenManageActions:
    type: array
    description: defaultForbiddenManageActions
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ADD
          description: Add
        - type: string
          enum:
            - EDIT
          description: Edit
        - type: string
          enum:
            - LIMITED_EDIT
          description: Limited Edit
        - type: string
          enum:
            - VIEW
          description: View
        - type: string
          enum:
            - DELETE
          description: Delete
        - type: string
          enum:
            - LEGACY_APPROVE
          description: Approve
        - type: string
          enum:
            - LEGACY_REQUEST_PROJECT
          description: Request
        - type: string
          enum:
            - CHANGE_STATUS
          description: Change Status
        - type: string
          enum:
            - EDIT_CUSTOMDATA
          description: Edit Custom Form
        - type: string
          enum:
            - EDIT_APPROVALPROCESS
          description: Edit Approval Process
        - type: string
          enum:
            - EDIT_ROLE_GROUP
          description: Change a User's Role or Group
        - type: string
          enum:
            - EDIT_ACCESSLEVEL
          description: Change a User's Access Level
        - type: string
          enum:
            - ADD_TASKS
          description: Add Tasks
        - type: string
          enum:
            - ADD_ISSUES
          description: Add Issues
        - type: string
          enum:
            - EDIT_ASSIGNMENTS
          description: Make Assignments
        - type: string
          enum:
            - ADD_DOCUMENTS
          description: Add Documents
        - type: string
          enum:
            - EDIT_FINANCE
          description: Manage Finance
        - type: string
          enum:
            - VIEW_FINANCE
          description: View Finance
        - type: string
          enum:
            - ADD_HOURS
          description: Log Hours
        - type: string
          enum:
            - ADD_EXPENSES
          description: Add Expenses
        - type: string
          enum:
            - SHARE
          description: Share
        - type: string
          enum:
            - PROOF
          description: Proof
        - type: string
          enum:
            - DOWNLOAD
          description: Download
        - type: string
          enum:
            - SHARE_SYSTEMWIDE
          description: Share System-wide
        - type: string
          enum:
            - VIEW_CONTACTINFO
          description: View Contact Info
        - type: string
          enum:
            - ATTACH_CUSTOMFORM
          description: Attach to Custom Data
        - type: string
          enum:
            - BUDGETING_INFORMATION
          description: Edit priorities and budget hours in the Planner
        - type: string
          enum:
            - PLANNED_HOURS_CONTOURING
          description: Update Planned Hours in the Workload Balancer
        - type: string
          enum:
            - ADD_TO_CUSTOM_FORMS
          description: Add to Custom Forms
        - type: string
          enum:
            - ADD_BOOKINGS
          description: Add Bookings
        - type: string
          enum:
            - VIEW_COST_RATES
          description: View Cost Rates
        - type: string
          enum:
            - VIEW_BILLING_RATES
          description: View Billing Rates
        - type: string
          enum:
            - VIEW_GENERAL_FINANCE
          description: View general finance
        - type: string
          enum:
            - EDIT_COST_RATES
          description: Manage Cost Rates
        - type: string
          enum:
            - EDIT_BILLING_RATES
          description: Manage Billing Rates
        - type: string
          enum:
            - EDIT_GENERAL_FINANCE
          description: Manage general finance
        - type: string
          enum:
            - EDIT_TEAMS_I_AM_ON
          description: Edit teams I am on
        - type: string
          enum:
            - EDIT_TEAMS_I_GROUP_ADMIN
          description: Edit teams in groups I manage (Group Admins only)
        - type: string
          enum:
            - EDIT_SYSTEMWIDE
          description: enum.actiontypeenum.editystemwide
    uniqueItems: true
  defaultForbiddenViewActions:
    type: array
    description: defaultForbiddenViewActions
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ADD
          description: Add
        - type: string
          enum:
            - EDIT
          description: Edit
        - type: string
          enum:
            - LIMITED_EDIT
          description: Limited Edit
        - type: string
          enum:
            - VIEW
          description: View
        - type: string
          enum:
            - DELETE
          description: Delete
        - type: string
          enum:
            - LEGACY_APPROVE
          description: Approve
        - type: string
          enum:
            - LEGACY_REQUEST_PROJECT
          description: Request
        - type: string
          enum:
            - CHANGE_STATUS
          description: Change Status
        - type: string
          enum:
            - EDIT_CUSTOMDATA
          description: Edit Custom Form
        - type: string
          enum:
            - EDIT_APPROVALPROCESS
          description: Edit Approval Process
        - type: string
          enum:
            - EDIT_ROLE_GROUP
          description: Change a User's Role or Group
        - type: string
          enum:
            - EDIT_ACCESSLEVEL
          description: Change a User's Access Level
        - type: string
          enum:
            - ADD_TASKS
          description: Add Tasks
        - type: string
          enum:
            - ADD_ISSUES
          description: Add Issues
        - type: string
          enum:
            - EDIT_ASSIGNMENTS
          description: Make Assignments
        - type: string
          enum:
            - ADD_DOCUMENTS
          description: Add Documents
        - type: string
          enum:
            - EDIT_FINANCE
          description: Manage Finance
        - type: string
          enum:
            - VIEW_FINANCE
          description: View Finance
        - type: string
          enum:
            - ADD_HOURS
          description: Log Hours
        - type: string
          enum:
            - ADD_EXPENSES
          description: Add Expenses
        - type: string
          enum:
            - SHARE
          description: Share
        - type: string
          enum:
            - PROOF
          description: Proof
        - type: string
          enum:
            - DOWNLOAD
          description: Download
        - type: string
          enum:
            - SHARE_SYSTEMWIDE
          description: Share System-wide
        - type: string
          enum:
            - VIEW_CONTACTINFO
          description: View Contact Info
        - type: string
          enum:
            - ATTACH_CUSTOMFORM
          description: Attach to Custom Data
        - type: string
          enum:
            - BUDGETING_INFORMATION
          description: Edit priorities and budget hours in the Planner
        - type: string
          enum:
            - PLANNED_HOURS_CONTOURING
          description: Update Planned Hours in the Workload Balancer
        - type: string
          enum:
            - ADD_TO_CUSTOM_FORMS
          description: Add to Custom Forms
        - type: string
          enum:
            - ADD_BOOKINGS
          description: Add Bookings
        - type: string
          enum:
            - VIEW_COST_RATES
          description: View Cost Rates
        - type: string
          enum:
            - VIEW_BILLING_RATES
          description: View Billing Rates
        - type: string
          enum:
            - VIEW_GENERAL_FINANCE
          description: View general finance
        - type: string
          enum:
            - EDIT_COST_RATES
          description: Manage Cost Rates
        - type: string
          enum:
            - EDIT_BILLING_RATES
          description: Manage Billing Rates
        - type: string
          enum:
            - EDIT_GENERAL_FINANCE
          description: Manage general finance
        - type: string
          enum:
            - EDIT_TEAMS_I_AM_ON
          description: Edit teams I am on
        - type: string
          enum:
            - EDIT_TEAMS_I_GROUP_ADMIN
          description: Edit teams in groups I manage (Group Admins only)
        - type: string
          enum:
            - EDIT_SYSTEMWIDE
          description: enum.actiontypeenum.editystemwide
    uniqueItems: true
  deliverableScoreCardID:
    type: string
    description: Deliverable Scorecard ID
    verbs: []
  deliverableSuccessScore:
    type: number
    description: Goal Success Score
    verbs: []
  deliverableSuccessScoreRatio:
    type: number
    description: deliverableSuccessScoreRatio
    verbs: []
  description:
    type: string
    description: The detailed description of this Project
    verbs:
      - GET
      - PUT
      - POST
  displayOrder:
    type: integer
    description: Display Order
    verbs: []
  durationExpression:
    type: string
    description: durationExpression
    verbs: []
  durationMinutes:
    type: integer
    description: Duration in Minutes
    verbs: []
  eac:
    type: number
    description: EAC
    verbs: []
  eacCalculationMethod:
    type: string
    description: EAC Calculation Method
    verbs: []
    oneOf:
      - type: string
        enum:
          - P
        description: Calculate at project level
      - type: string
        enum:
          - T
        description: Roll up from tasks/subtasks
  enableAutoBaselines:
    type: boolean
    description: Enable Auto Baselines
    verbs: []
  enteredByID:
    type: string
    description: The ID of the user who entered this Project
    verbs:
      - GET
  entryDate:
    type: string
    description: The date in which this project was created
    verbs:
      - GET
  estCompletionDate:
    type: string
    description: The date in which this project is estimated to be complete. This is a calculated field.
    verbs:
      - GET
  estStartDate:
    type: string
    description: The data in which we expect to start working on this project
    verbs:
      - GET
  extRefID:
    type: string
    description: External Reference ID
    verbs: []
  filterHourTypes:
    type: boolean
    description: Filter Hour Types
    verbs: []
  financeLastUpdateDate:
    type: string
    description: Last Finance Update Date
    verbs: []
  fixedCost:
    type: number
    description: Fixed Cost
    verbs: []
  fixedEndDate:
    type: string
    description: Fixed End Date
    verbs: []
  fixedRevenue:
    type: number
    description: Fixed Revenue
    verbs: []
  fixedStartDate:
    type: string
    description: Fixed Start Date
    verbs: []
  goalHierarchy:
    type: string
    description: goalHierarchy
    verbs: []
  goalsString:
    type: string
    description: goalsString
    verbs: []
  groupID:
    type: string
    description: The ID of the group associated with this Project. Users can belong to Groups.
    verbs:
      - GET
      - PUT
      - POST
  hasBookings:
    type: boolean
    description: hasBookings
    verbs: []
  hasBudgetConflict:
    type: boolean
    description: Has Budget Conflict
    verbs: []
  hasCalcError:
    type: boolean
    description: Has Calc Error
    verbs: []
  hasCompletionConstraint:
    type: boolean
    description: hasCompletionConstraint
    verbs: []
  hasDocuments:
    type: boolean
    description: Has Documents
    verbs: []
  hasExpenses:
    type: boolean
    description: Has Expenses
    verbs: []
  hasMessages:
    type: boolean
    description: Has Messages
    verbs: []
  hasNotes:
    type: boolean
    description: Has Notes
    verbs: []
  hasRateOverride:
    type: boolean
    description: Has Billing Rate Override
    verbs: []
  hasResolvables:
    type: boolean
    description: Has Resolvables
    verbs: []
  hasStartConstraint:
    type: boolean
    description: hasStartConstraint
    verbs: []
  hasTimedNotifications:
    type: boolean
    description: Has Reminder Notifications
    verbs: []
  hasTimelineException:
    type: boolean
    description: hasTimelineException
    verbs: []
  ID:
    type: string
    description: The unique ID or identifier of this project
    verbs:
      - GET
      - PUT
      - POST
      - DELETE
  isOriginalPlannedHoursSet:
    type: boolean
    description: isOriginalPlannedHoursSet
    verbs: []
  isProjectDead:
    type: boolean
    description: isProjectDead
    verbs: []
  isStatusComplete:
    type: boolean
    description: isStatusComplete
    verbs: []
  lastCalcDate:
    type: string
    description: Last Calc Date
    verbs: []
  lastConditionNoteID:
    type: string
    description: Last Condition Note ID
    verbs: []
  lastNoteID:
    type: string
    description: Last Note ID
    verbs: []
  lastUpdateDate:
    type: string
    description: The date of the last update to this Project
    verbs:
      - GET
  lastUpdatedByID:
    type: string
    description: The ID of the user who last updated this Project
    verbs:
      - GET
  levelingMode:
    type: string
    description: Leveling Mode
    verbs: []
    oneOf:
      - type: string
        enum:
          - AUT
        description: Automatic
      - type: string
        enum:
          - MAN
        description: Manual
  lucidMigrationDate:
    type: string
    description: lucidMigrationDate
    verbs: []
  milestonePathID:
    type: string
    description: Milestone Path ID
    verbs: []
  name:
    type: string
    description: The name of this Project
    verbs:
      - GET
      - PUT
      - POST
  nextAutoBaselineDate:
    type: string
    description: Next Date
    verbs: []
  numberOfLinkedGoals:
    type: integer
    description: Number of linked Goals
    verbs: []
  numberOfOpenRisks:
    type: integer
    description: Number of Open Risks
    verbs: []
  numberOpenOpTasks:
    type: integer
    description: Number of Open Issues
    verbs: []
  olv:
    type: integer
    description: Optimization Log Value
    verbs: []
  optimizationScore:
    type: integer
    description: Optimization Score
    verbs: []
  originalWorkRequired:
    type: integer
    description: Original Planned Hours
    verbs: []
  ownerID:
    type: string
    description: The person who owns or is responsible for this Project (The Project Owner)
    verbs:
      - GET
      - PUT
      - POST
  ownerPrivileges:
    type: array
    description: Owner Privileges
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - APT
          description: Require time to be approved for this project
    uniqueItems: true
  pendingCalculation:
    type: string
    description: pendingCalculation
    verbs: []
  pendingUpdateMethods:
    type: array
    description: pendingUpdateMethods
    verbs: []
    items:
      type: string
    uniqueItems: true
  percentComplete:
    type: number
    description: The current percent complete of this Project
    verbs:
      - GET
  performanceIndexMethod:
    type: string
    description: Performance Index Method
    verbs: []
    oneOf:
      - type: string
        enum:
          - H
        description: Hour-Based
      - type: string
        enum:
          - C
        description: Cost-Based
  personal:
    type: boolean
    description: Personal
    verbs: []
  plannedBenefit:
    type: number
    description: Planned Benefit
    verbs: []
  plannedBillableExpenseCost:
    type: number
    description: plannedBillableExpenseCost
    verbs: []
  plannedCompletionDate:
    type: string
    description: The date in which we planned this project to complete - this is not the actual completion date
    verbs:
      - GET
  plannedCost:
    type: number
    description: Planned Cost
    verbs: []
  plannedDateAlignment:
    type: integer
    description: Planned Date Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Will be done after the planned completion date
      - type: integer
        enum:
          - 1
        description: Will be done on the planned completion date
      - type: integer
        enum:
          - 2
        description: Will be done before the planned completion date
      - type: integer
        enum:
          - 3
        description: No set planned date
  plannedExpenseCost:
    type: number
    description: Planned Expense Cost
    verbs: []
  plannedHoursAlignment:
    type: integer
    description: Planned Hours Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: More hours than planned
      - type: integer
        enum:
          - 1
        description: Same hours as planned
      - type: integer
        enum:
          - 2
        description: Less hours than planned
      - type: integer
        enum:
          - 3
        description: No planned hours set
  plannedLaborCost:
    type: number
    description: Planned Labor Cost
    verbs: []
  plannedLaborRevenue:
    type: number
    description: Planned Labor Revenue
    verbs: []
  plannedNonBillableExpenseCost:
    type: number
    description: plannedNonBillableExpenseCost
    verbs: []
  plannedNonLaborCost:
    type: number
    description: Planned Non-Labor Cost
    verbs: []
  plannedNonLaborRevenue:
    type: number
    description: Planned Non-Labor Revenue
    verbs: []
  plannedRevenue:
    type: number
    description: Planned Revenue
    verbs: []
  plannedRiskCost:
    type: number
    description: Planned Risk Cost
    verbs: []
  plannedStartDate:
    type: string
    description: The date in which we plan to start work on this Project
    verbs:
      - GET
      - PUT
      - POST
  plannedValue:
    type: number
    description: Planned Value
    verbs: []
  popAccountID:
    type: string
    description: Pop Account ID
    verbs: []
  portfolioID:
    type: string
    description: The ID of the portfolio this project belongs to
    verbs:
      - GET
      - PUT
      - POST
  portfolioPriority:
    type: integer
    description: Portfolio Priority
    verbs: []
  previousStatus:
    type: string
    description: Previous status
    verbs: []
    oneOf:
      - type: string
        enum:
          - CUR
        description: Current
      - type: string
        enum:
          - ONH
        description: On Hold
      - type: string
        enum:
          - PLN
        description: Planning
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - DED
        description: Dead
      - type: string
        enum:
          - REQ
        description: Requested
      - type: string
        enum:
          - APR
        description: Approved
      - type: string
        enum:
          - REJ
        description: Rejected
      - type: string
        enum:
          - IDA
        description: Idea
  previousStatusLabel:
    type: string
    description: previousStatusLabel
    verbs: []
  priority:
    type: integer
    description: "Priority.  Use the enum values, not the descriptions.  The only accepted values are: 0, 1, 2, 3, 4."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: integer
        enum:
          - 0
        description: None
      - type: integer
        enum:
          - 1
        description: Low
      - type: integer
        enum:
          - 2
        description: Normal
      - type: integer
        enum:
          - 3
        description: High
      - type: integer
        enum:
          - 4
        description: Urgent
  privateRateCardID:
    type: string
    description: privateRateCardID
    verbs: []
  programID:
    type: string
    description: The ID or the Program this project is associated with
    verbs:
      - GET
      - PUT
      - POST
  progressStatus:
    type: string
    description: Progress Status
    verbs: []
    oneOf:
      - type: string
        enum:
          - LT
        description: Late
      - type: string
        enum:
          - "ON"
        description: On Time
      - type: string
        enum:
          - BH
        description: Behind
      - type: string
        enum:
          - RK
        description: At Risk
  projectBudgetedCost:
    type: number
    description: This is a calculated field representing the total budgeted cost of this project
    verbs:
      - GET
  projectedCompletionDate:
    type: string
    description: Calculated field represented the projected completion date of this project
    verbs:
      - GET
  projectedStartDate:
    type: string
    description: Calculated field representing the projected start date of this project
    verbs:
      - GET
  projectNetValue:
    type: number
    description: Project Net Value
    verbs: []
  projectRoi:
    type: number
    description: Project ROI
    verbs: []
  queueDefID:
    type: string
    description: Queue Definition ID
    verbs: []
  referenceNumber:
    type: integer
    description: A system generated value used to reference this project
    verbs:
      - GET
  rejectionIssueID:
    type: string
    description: Rejection Issue ID
    verbs: []
  remainingCost:
    type: number
    description: Remaining Cost
    verbs: []
  remainingRevenue:
    type: number
    description: Remaining Revenue
    verbs: []
  remainingRiskCost:
    type: number
    description: Remaining Risk Cost
    verbs: []
  resourcePlannerBudgetedHours:
    type: number
    description: Resource Planner Budgeted Hours
    verbs: []
  resourcePlannerBudgetedLaborCost:
    type: number
    description: Resource Planner Budgeted Labor Cost
    verbs: []
  resourcePoolID:
    type: string
    description: Legacy Resource Pool ID
    verbs: []
  risk:
    type: integer
    description: Risk
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: None
      - type: integer
        enum:
          - 1
        description: Very Low
      - type: integer
        enum:
          - 2
        description: Low
      - type: integer
        enum:
          - 3
        description: Medium
      - type: integer
        enum:
          - 4
        description: High
      - type: integer
        enum:
          - 5
        description: Very High
  riskPerformanceIndex:
    type: number
    description: RPI
    verbs: []
  roi:
    type: number
    description: ROI
    verbs: []
  rootGroupID:
    type: string
    description: rootGroupID
    verbs: []
  scenarioPlannerBudgetedLaborCost:
    type: number
    description: scenarioPlannerBudgetedLaborCost
    verbs: []
  scheduleID:
    type: string
    description: Schedule ID
    verbs: []
  scheduleMode:
    type: string
    description: Schedule Mode
    verbs: []
    oneOf:
      - type: string
        enum:
          - S
        description: Start Date
      - type: string
        enum:
          - C
        description: Completion Date
  securityAncestorsDisabled:
    type: boolean
    description: securityAncestorsDisabled
    verbs: []
  selectedOnPortfolioOptimizer:
    type: boolean
    description: Selected on Portfolio Optimizer
    verbs: []
  showCondition:
    type: boolean
    description: showCondition
    verbs: []
  showStatus:
    type: boolean
    description: showStatus
    verbs: []
  spi:
    type: number
    description: Schedule Performance Index
    verbs: []
  sponsorID:
    type: string
    description: Sponsor ID
    verbs: []
  status:
    type: string
    description: "Status.  Use the enum values, not the descriptions.  The only accepted values are: CUR, ONH, PLN, CPL, DED, REQ, APR, REJ, IDA."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - CUR
        description: Current
      - type: string
        enum:
          - ONH
        description: On Hold
      - type: string
        enum:
          - PLN
        description: Planning
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - DED
        description: Dead
      - type: string
        enum:
          - REQ
        description: Requested
      - type: string
        enum:
          - APR
        description: Approved
      - type: string
        enum:
          - REJ
        description: Rejected
      - type: string
        enum:
          - IDA
        description: Idea
  statusLabel:
    type: string
    description: Status Label
    verbs: []
  statusUpdate:
    type: string
    description: Status Update
    verbs: []
  storiesWithoutPoints:
    type: integer
    description: Tasks Without Points
    verbs: []
  storyPointsPercentComplete:
    type: number
    description: Percent Complete (Story Points)
    verbs: []
  submittedByID:
    type: string
    description: The ID of the user who submitted this Project
    verbs:
      - GET
  subscribersListString:
    type: string
    description: subscribersListString
    verbs: []
  summaryCompletionType:
    type: string
    description: summaryCompletionType
    verbs: []
    oneOf:
      - type: string
        enum:
          - MAN
        description: Manual
      - type: string
        enum:
          - AUT
        description: Automatic
  syncBurndownDate:
    type: string
    description: syncBurndownDate
    verbs: []
  teamID:
    type: string
    description: The ID of the team associated with this Project
    verbs: []
  templateID:
    type: string
    description: The ID of the template that was used to create this project
    verbs:
      - GET
      - POST
  totalHours:
    type: number
    description: Total Hours
    verbs: []
  totalOpTaskCount:
    type: integer
    description: Total Issue Count
    verbs: []
  totalStoryPoints:
    type: number
    description: Total Story Points
    verbs: []
  totalTaskCount:
    type: integer
    description: Total Task Count
    verbs: []
  updateType:
    type: string
    description: Update Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - AUTO
        description: Automatic and On Change
      - type: string
        enum:
          - EVNT
        description: Change Only
      - type: string
        enum:
          - ATYO
        description: Automatic Only
      - type: string
        enum:
          - MANU
        description: Manual Only
  URL:
    type: string
    description: URL
    verbs: []
  useDateEffectiveExchangeRates:
    type: boolean
    description: useDateEffectiveExchangeRates
    verbs: []
  userTimeOff:
    type: string
    description: User Time Off
    verbs: []
    oneOf:
      - type: string
        enum:
          - CUTO
        description: Consider user time off in task durations
      - type: string
        enum:
          - IUTO
        description: Ignore user time off in task durations
  version:
    type: integer
    description: Version
    verbs: []
  work:
    type: number
    description: Planned Hours
    verbs: []
  workRequired:
    type: integer
    description: Planned Hours
    verbs: []
  workRequiredExpression:
    type: string
    description: workRequiredExpression
    verbs: []

--- TASK METADATA ---
domain_knowledge: |
  Tasks always belong to a project, so they always have the field projectID even when updating a task or making an assignment. 
  To update existing tasks, you must include the taskID in the request. 
  Tasks may also have subtasks, in which case they have a parent task (parentID) and a projectID.
  Tasks frequently have an assignment to a user, team, or role.
  Tasks can have both a plannedCompletionDate and a actualCompletionDate - if you want to know the actual date something was completed, use actualCompletionDate.
  Assignments can be added when creating an task by adding appropriate ids to the assignments array.
  A task name is always required.
  To set a Due Date, set the taskConstraint to MFO and the constraintDate to the desired date.  
  To set a Start Date, set the taskConstraint to MSO and the constraintDate to the desired date.
  Default statuses include: New, In Progress, and Complete
  Tasks include a taskNumber.  The task number shows the order the task appears in a list as shown to the user.  The user may referer to a task by its task number instead of its name.
  If a user refers to a task by its number insetead of its name, then it will need to be search by taskNumber instead of name to find the task id.  i.e. I want to rename task 3 to "New Task Name"
fields:
  accessorIDs:
    type: array
    description: accessorIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  actualBillableExpenseCost:
    type: number
    description: actualBillableExpenseCost
    verbs: []
  actualCompletionDate:
    type: string
    description: The date in which this task was actually completed
    verbs:
      - GET
      - PUT
      - POST
  actualCost:
    type: number
    description: Actual Cost
    verbs:
      - GET
      - PUT
      - POST
  actualDuration:
    type: number
    description: Actual Duration
    verbs: []
  actualDurationMinutes:
    type: integer
    description: Actual Duration Minutes
    verbs: []
  actualExpenseCost:
    type: number
    description: Actual Expense Cost
    verbs: []
  actualLaborCost:
    type: number
    description: Actual Labor Cost
    verbs: []
  actualLaborRevenue:
    type: number
    description: Actual Labor Revenue
    verbs: []
  actualNonBillableExpenseCost:
    type: number
    description: actualNonBillableExpenseCost
    verbs: []
  actualNonLaborCost:
    type: number
    description: Actual Non-Labor Cost
    verbs: []
  actualNonLaborRevenue:
    type: number
    description: Actual Non-Labor Revenue
    verbs: []
  actualRevenue:
    type: number
    description: Actual Revenue
    verbs: []
  actualStartDate:
    type: string
    description: Actual Start Date
    verbs:
      - GET
      - PUT
      - POST
  actualWork:
    type: number
    description: Actual Hours
    verbs: []
  actualWorkRequired:
    type: integer
    description: Actual Hours
    verbs: []
  actualWorkRequiredDouble:
    type: number
    description: actualWorkRequiredDouble
    verbs: []
  approvalCompletionDate:
    type: string
    description: Approval Path Completion Date
    verbs: []
  approvalEstStartDate:
    type: string
    description: Approval Estimated Start Date
    verbs: []
  approvalPlannedStartDate:
    type: string
    description: Approval Planned Start Date
    verbs: []
  approvalPlannedStartDay:
    type: number
    description: approvalPlannedStartDay
    verbs: []
  approvalProcessID:
    type: string
    description: Approval Processes ID
    verbs: []
  approvalProjectedStartDate:
    type: string
    description: Approval Projected Start Date
    verbs: []
  approvalStartDate:
    type: string
    description: Approval Path Start Date
    verbs: []
  approversString:
    type: string
    description: approversString
    verbs: []
  assignments:
    type: array
    description: User, Role, or Team ID  assigned to this object for use when more than one assignment is needed
    verbs:
      - POST
      - GET
      - PUT
    items:
      type: object
      properties:
        assignedToID:
          type: string
          description: User, Role, or Team ID  assigned to this object
          verbs:
            - GET
            - POST
            - PUT
  assignedToID:
    type: string
    description: Assigned To ID
    verbs:
      - POST
      - GET
      - PUT
  assignmentsListString:
    type: string
    description: assignmentsListString
    verbs: []
  auditNote:
    type: string
    description: auditNote
    verbs: []
  auditTypes:
    type: array
    description: Audit Types
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ST
          description: Status Change
        - type: string
          enum:
            - AA
          description: Attachment Action
        - type: string
          enum:
            - SC
          description: Scope Change
        - type: string
          enum:
            - "NO"
          description: Note
        - type: string
          enum:
            - GE
          description: General Edit
        - type: string
          enum:
            - CM
          description: Combined Entry
        - type: string
          enum:
            - ER
          description: Error Entry
        - type: string
          enum:
            - SB
          description: Subscription Change
    uniqueItems: true
  auditUserIDs:
    type: array
    description: auditUserIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  backlogOrder:
    type: integer
    description: Order
    verbs: []
  backlogParent:
    type: string
    description: Parent
    verbs: []
  bcwp:
    type: number
    description: BCWP
    verbs: []
  bcws:
    type: number
    description: BCWS
    verbs: []
  billingAmount:
    type: number
    description: Cap or Fixed Amount
    verbs: []
  billingRecordID:
    type: string
    description: Billing Record ID
    verbs: []
  canStart:
    type: boolean
    description: Can Start
    verbs:
      - GET
      - PUT
      - POST
  categoryID:
    type: string
    description: Category ID
    verbs: []
  childrenIds:
    type: array
    description: childrenIds
    verbs: []
    items:
      type: string
    uniqueItems: true
  color:
    type: string
    description: Color
    verbs: []
  commitDate:
    type: string
    description: Commit Date
    verbs: []
  commitDateRange:
    type: string
    description: commitDateRange
    verbs: []
  completionPendingDate:
    type: string
    description: Completion Pending Date
    verbs: []
  condition:
    type: integer
    description: "Condition.  Use the enum values, not the descriptions.  The only accepted values are: 0, 1, 2."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: integer
        enum:
          - 0
        description: Going Smoothly
      - type: integer
        enum:
          - 1
        description: Some Concerns
      - type: integer
        enum:
          - 2
        description: Major Roadblocks
  constraintDate:
    type: string
    description: Constraint Date
    verbs:
      - GET
      - PUT
      - POST
  convertedOpTaskEntryDate:
    type: string
    description: Converted Issue Entry Date
    verbs: []
  convertedOpTaskID:
    type: string
    description: convertedOpTaskID
    verbs: []
  convertedOpTaskName:
    type: string
    description: Converted Issue Name
    verbs: []
  convertedOpTaskOriginatorID:
    type: string
    description: convertedOpTaskOriginatorID
    verbs: []
  costAmount:
    type: number
    description: Cost Amount
    verbs: []
  costType:
    type: string
    description: Cost Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - NOC
        description: No Cost
      - type: string
        enum:
          - FHR
        description: Fixed Hourly
      - type: string
        enum:
          - UHR
        description: User Hourly
      - type: string
        enum:
          - RHC
        description: Role Hourly
  cpi:
    type: number
    description: Cost Performance Index
    verbs: []
  csi:
    type: number
    description: Cost Schedule Performance Index
    verbs: []
  currentApprovalStepID:
    type: string
    description: Current Approval Stage ID
    verbs: []
  customerID:
    type: string
    description: Customer ID
    verbs: []
  customFormsDescriptions:
    type: string
    description: customFormsDescriptions
    verbs: []
  customFormsGroupIDs:
    type: string
    description: customFormsGroupIDs
    verbs: []
  customFormsIDs:
    type: string
    description: customFormsIDs
    verbs:
      - GET
  customFormsNames:
    type: string
    description: customFormsNames
    verbs:
      - GET
  daysLate:
    type: integer
    description: Days Late
    verbs: []
  description:
    type: string
    description: Description
    verbs:
      - GET
      - PUT
      - POST
  duration:
    type: number
    description: Duration.  Use this field to set the duration of the task.  The duration is the amount of time it will take to complete the task.  Always set durationUnit when setting duration.
    verbs:
      - GET
      - PUT
      - POST
  durationExpression:
    type: string
    description: durationExpression
    verbs: []
  durationMinutes:
    type: integer
    description: Duration in Minutes
    verbs: []
  durationType:
    type: string
    description: Duration Type
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - W
        description: Calculated Work
      - type: string
        enum:
          - D
        description: Effort Driven
      - type: string
        enum:
          - A
        description: Calculated Assignment
      - type: string
        enum:
          - S
        description: Simple
  durationUnit:
    type: string
    description: Duration Unit
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - M
        description: Minutes
      - type: string
        enum:
          - H
        description: Hours
      - type: string
        enum:
          - D
        description: Days
      - type: string
        enum:
          - W
        description: Weeks
      - type: string
        enum:
          - T
        description: Months
      - type: string
        enum:
          - EM
        description: Elapsed Minutes
      - type: string
        enum:
          - EH
        description: Elapsed Hours
      - type: string
        enum:
          - ED
        description: Elapsed Days
      - type: string
        enum:
          - EW
        description: Elapsed Weeks
      - type: string
        enum:
          - ET
        description: Elapsed Months
  eac:
    type: number
    description: EAC
    verbs: []
  enteredByID:
    type: string
    description: Entered By ID
    verbs:
      - GET
  entryDate:
    type: string
    description: The date in which this task was created or entered
    verbs:
      - GET
  estCompletionDate:
    type: string
    description: Estimated Due Date
    verbs: []
  estimate:
    type: number
    description: Estimate
    verbs:
      - GET
  estStartDate:
    type: string
    description: Estimated Start Date
    verbs: []
  extRefID:
    type: string
    description: External Reference ID
    verbs: []
  groupID:
    type: string
    description: Group ID
    verbs:
      - GET
      - PUT
      - POST
  handoffDate:
    type: string
    description: Handoff Date
    verbs: []
  hasBookings:
    type: boolean
    description: hasBookings
    verbs: []
  hasCompletionConstraint:
    type: boolean
    description: hasCompletionConstraint
    verbs: []
  hasDocuments:
    type: boolean
    description: Has Documents
    verbs: []
  hasExpenses:
    type: boolean
    description: Has Expenses
    verbs: []
  hasMessages:
    type: boolean
    description: Has Messages
    verbs: []
  hasNotes:
    type: boolean
    description: Has Notes
    verbs: []
  hasResolvables:
    type: boolean
    description: Has Resolvables
    verbs: []
  hasStartConstraint:
    type: boolean
    description: hasStartConstraint
    verbs: []
  hasTimedNotifications:
    type: boolean
    description: Has Reminder Notifications
    verbs: []
  hoursPerPoint:
    type: number
    description: hoursPerPoint
    verbs: []
  ID:
    type: string
    description: Task ID
    verbs:
      - GET
      - PUT
      - POST
      - DELETE
  indent:
    type: integer
    description: Indent
    verbs: []
  isAgile:
    type: boolean
    description: Agile
    verbs: []
  isCritical:
    type: boolean
    description: Is Critical
    verbs: []
  isDurationLocked:
    type: boolean
    description: Duration Locked
    verbs: []
  isLevelingExcluded:
    type: boolean
    description: Exclude from Leveling
    verbs: []
  isReady:
    type: boolean
    description: Ready
    verbs: []
  isStatusComplete:
    type: boolean
    description: isStatusComplete
    verbs: []
  isWorkRequiredLocked:
    type: boolean
    description: Worked Locked
    verbs: []
  iterationID:
    type: string
    description: Iteration ID
    verbs:
      - GET
      - PUT
      - POST
  kanbanBoardID:
    type: string
    description: Kanban Board ID
    verbs: []
  kanbanFlag:
    type: string
    description: Kanban Flag
    verbs: []
    oneOf:
      - type: string
        enum:
          - ONTRACK
        description: On Track
      - type: string
        enum:
          - ISBLOCKED
        description: Is Blocked
      - type: string
        enum:
          - READYTOPULL
        description: Ready to Pull
  lastConditionNoteID:
    type: string
    description: Last Condition Note ID
    verbs: []
  lastNoteID:
    type: string
    description: Last Note ID
    verbs: []
  lastUpdateDate:
    type: string
    description: Last Update Date
    verbs:
      - GET
  lastUpdatedByID:
    type: string
    description: Last Updated By ID
    verbs:
      - GET
  levelingStartDelay:
    type: number
    description: Leveling Delay
    verbs: []
  levelingStartDelayExpression:
    type: string
    description: levelingStartDelayExpression
    verbs: []
  levelingStartDelayMinutes:
    type: integer
    description: levelingStartDelayMinutes
    verbs: []
  milestoneID:
    type: string
    description: Milestone ID
    verbs: []
  name:
    type: string
    description: Name
    verbs:
      - GET
      - PUT
      - POST
  numberOfChildren:
    type: integer
    description: Number of Children
    verbs: []
  numberOpenOpTasks:
    type: integer
    description: Number of Open Issues
    verbs: []
  olv:
    type: integer
    description: Optimization Log Value
    verbs: []
  originalActualWorkRequired:
    type: integer
    description: originalActualWorkRequired
    verbs: []
  originalActualWorkRequiredDouble:
    type: number
    description: originalActualWorkRequiredDouble
    verbs: []
  originalDuration:
    type: integer
    description: Original Duration
    verbs: []
  originalWorkRequired:
    type: integer
    description: Original Planned Hours
    verbs: []
  parentID:
    type: string
    description: Parent ID
    verbs:
      - GET
      - PUT
      - POST
  parentLag:
    type: number
    description: Parent Lag
    verbs: []
  parentLagType:
    type: string
    description: Parent Lag Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - d
        description: Days
      - type: string
        enum:
          - c
        description: Calendar Days
      - type: string
        enum:
          - p
        description: Percent
      - type: string
        enum:
          - w
        description: Day of Week
      - type: string
        enum:
          - k
        description: Day of Week (Non-Zero)
  pendingCalculation:
    type: string
    description: pendingCalculation
    verbs: []
  pendingPredecessors:
    type: boolean
    description: Pending Predecessors
    verbs: []
  pendingUpdateMethods:
    type: array
    description: pendingUpdateMethods
    verbs: []
    items:
      type: string
    uniqueItems: true
  percentComplete:
    type: number
    description: Percent Complete
    verbs:
      - GET
      - PUT
      - POST
  personal:
    type: boolean
    description: Personal
    verbs:
      - GET
      - POST
  plannedBillableExpenseCost:
    type: number
    description: plannedBillableExpenseCost
    verbs: []
  plannedCompletionDate:
    type: string
    description: The date in which this task is planned or estimated to be completed
    verbs:
      - GET
  plannedCompletionDateExpression:
    type: string
    description: plannedCompletionDateExpression
    verbs: []
  plannedCost:
    type: number
    description: Planned Cost
    verbs: []
  plannedDateAlignment:
    type: integer
    description: Planned Date Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Will be done after the planned completion date
      - type: integer
        enum:
          - 1
        description: Will be done on the planned completion date
      - type: integer
        enum:
          - 2
        description: Will be done before the planned completion date
      - type: integer
        enum:
          - 3
        description: No set planned date
  plannedDuration:
    type: number
    description: Planned Duration
    verbs:
      - GET
      - PUT
      - POST
  plannedDurationMinutes:
    type: integer
    description: Planned Duration Minutes
    verbs: []
  plannedExpenseCost:
    type: number
    description: Planned Expense Cost
    verbs: []
  plannedHoursAlignment:
    type: integer
    description: Planned Hours Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: More hours than planned
      - type: integer
        enum:
          - 1
        description: Same hours as planned
      - type: integer
        enum:
          - 2
        description: Less hours than planned
      - type: integer
        enum:
          - 3
        description: No planned hours set
  plannedLaborCost:
    type: number
    description: Planned Labor Cost
    verbs: []
  plannedLaborRevenue:
    type: number
    description: Planned Labor Revenue
    verbs: []
  plannedNonBillableExpenseCost:
    type: number
    description: plannedNonBillableExpenseCost
    verbs: []
  plannedNonLaborCost:
    type: number
    description: Planned Non-Labor Cost
    verbs: []
  plannedNonLaborRevenue:
    type: number
    description: Planned Non-Labor Revenue
    verbs: []
  plannedRevenue:
    type: number
    description: Planned Revenue
    verbs: []
  plannedStartDate:
    type: string
    description: Planned Start Date
    verbs: []
  plannedStartDateExpression:
    type: string
    description: plannedStartDateExpression
    verbs: []
  predecessorExpression:
    type: string
    description: predecessorExpression
    verbs: []
  previousStatus:
    type: string
    description: Previous status
    verbs: []
    oneOf:
      - type: string
        enum:
          - NEW
        description: New
      - type: string
        enum:
          - INP
        description: In Progress
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - CPA
        description: Complete - Pending Approval
      - type: string
        enum:
          - CPI
        description: Complete - Pending Issues
  previousStatusLabel:
    type: string
    description: previousStatusLabel
    verbs: []
  priority:
    type: integer
    description: "Priority.  Use the enum value, not the description.  The only accepted values are: 0, 1, 2, 3, 4."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: integer
        enum:
          - 0
        description: None
      - type: integer
        enum:
          - 1
        description: Low
      - type: integer
        enum:
          - 2
        description: Normal
      - type: integer
        enum:
          - 3
        description: High
      - type: integer
        enum:
          - 4
        description: Urgent
  priorityColor:
    type: string
    description: priorityColor
    verbs: []
  progressStatus:
    type: string
    description: Progress Status
    verbs: []
    oneOf:
      - type: string
        enum:
          - LT
        description: Late
      - type: string
        enum:
          - "ON"
        description: On Time
      - type: string
        enum:
          - BH
        description: Behind
      - type: string
        enum:
          - RK
        description: At Risk
  projectedCompletionDate:
    type: string
    description: Projected Completion Date
    verbs: []
  projectedDurationMinutes:
    type: integer
    description: Projected Duration Minutes
    verbs: []
  projectedStartDate:
    type: string
    description: Projected Start Date
    verbs:
      - GET
      - PUT
      - POST
  projectedWorkPerDayList:
    type: array
    description: projectedWorkPerDayList
    verbs: []
    items:
      type: string
    uniqueItems: true
  projectID:
    type: string
    description: Project ID. Required when creating or updating a task.
    verbs:
      - GET
      - PUT
      - POST
  recurrenceNumber:
    type: integer
    description: Recurrence Number
    verbs: []
  recurrenceRuleID:
    type: string
    description: Recurrence Rule ID
    verbs: []
  referenceNumber:
    type: integer
    description: Reference Number
    verbs:
      - GET
  rejectionIssueID:
    type: string
    description: Rejection Issue ID
    verbs: []
  remainingDurationMinutes:
    type: integer
    description: Remaining Duration Minutes
    verbs: []
  resourceScope:
    type: string
    description: Resource Scope
    verbs: []
    oneOf:
      - type: string
        enum:
          - NONE
        description: None
      - type: string
        enum:
          - GLOB
        description: Global
  revenueType:
    type: string
    description: Revenue Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - NOB
        description: Not Billable
      - type: string
        enum:
          - UHR
        description: User Hourly
      - type: string
        enum:
          - RHR
        description: Role Hourly
      - type: string
        enum:
          - FHR
        description: Fixed Hourly
      - type: string
        enum:
          - UHC
        description: User Hourly w/Cap
      - type: string
        enum:
          - RHC
        description: Role Hourly w/Cap
      - type: string
        enum:
          - UHF
        description: User Hourly Plus Fixed
      - type: string
        enum:
          - RHF
        description: Role Hourly Plus Fixed
      - type: string
        enum:
          - FRV
        description: Fixed Revenue
  roleID:
    type: string
    description: Role ID
    verbs: []
  rootGroupID:
    type: string
    description: rootGroupID
    verbs: []
  securityAncestorsDisabled:
    type: boolean
    description: securityAncestorsDisabled
    verbs: []
  securityRootID:
    type: string
    description: securityRootID
    verbs: []
  securityRootObjCode:
    type: string
    description: securityRootObjCode
    verbs: []
  showCommitDate:
    type: boolean
    description: showCommitDate
    verbs: []
  showCondition:
    type: boolean
    description: showCondition
    verbs: []
  showStatus:
    type: boolean
    description: showStatus
    verbs: []
  slackDate:
    type: string
    description: Slack Date
    verbs: []
  spi:
    type: number
    description: Schedule Performance Index
    verbs: []
  status:
    type: string
    description: "Status.  Use the enum value, not the description.  The only accepted values are: NEW, INP, CPL, CPA, CPI."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - NEW
        description: New
      - type: string
        enum:
          - INP
        description: In Progress
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - CPA
        description: Complete - Pending Approval
      - type: string
        enum:
          - CPI
        description: Complete - Pending Issues
  statusEquatesWith:
    type: string
    description: Status Equates With
    verbs: []
  statusLabel:
    type: string
    description: Status Label
    verbs: []
  statusUpdate:
    type: string
    description: Status Update
    verbs: []
  storyBoardOrder:
    type: integer
    description: storyBoardOrder
    verbs: []
  storyPoints:
    type: number
    description: Story Points
    verbs:
      - GET
      - PUT
      - POST
  submittedByID:
    type: string
    description: Submitted By ID
    verbs: []
  subscribersListString:
    type: string
    description: subscribersListString
    verbs: []
  taskConstraint:
    type: string
    description: Task Constraint
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - FIXT
        description: Fixed Dates
      - type: string
        enum:
          - MSO
        description: Must Start On
      - type: string
        enum:
          - MFO
        description: Must Finish On
      - type: string
        enum:
          - ASAP
        description: As Soon As Possible
      - type: string
        enum:
          - ALAP
        description: As Late As Possible
      - type: string
        enum:
          - EAT
        description: Earliest Available Time
      - type: string
        enum:
          - LAT
        description: Latest Available Time
      - type: string
        enum:
          - SNLT
        description: Start No Later Than
      - type: string
        enum:
          - SNET
        description: Start No Earlier Than
      - type: string
        enum:
          - FNLT
        description: Finish No Later Than
      - type: string
        enum:
          - FNET
        description: Finish No Earlier Than
  taskNumber:
    type: integer
    description: Task Number
    verbs:
      - GET
      - PUT
      - POST
  taskNumberPredecessorString:
    type: string
    description: taskNumberPredecessorString
    verbs: []
  teamID:
    type: string
    description: The ID of the team assigned to this Task
    verbs:
      - GET
      - PUT
      - POST
  templateTaskID:
    type: string
    description: Template Task ID
    verbs: []
  trackingMode:
    type: string
    description: Tracking Mode
    verbs: []
    oneOf:
      - type: string
        enum:
          - USER
        description: User Must Update
      - type: string
        enum:
          - ONTM
        description: Assume On Time
      - type: string
        enum:
          - IGNR
        description: Ignore Late Warnings
      - type: string
        enum:
          - AUTO
        description: Auto Complete
      - type: string
        enum:
          - PRED
        description: Predecessor
  URL:
    type: string
    description: URL
    verbs: []
  version:
    type: integer
    description: Version
    verbs: []
  wbs:
    type: string
    description: Work Breakdown Structure
    verbs: []
  work:
    type: number
    description: Planned Hours
    verbs: []
  workEffort:
    type: integer
    description: Work Effort
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 1
        description: Small
      - type: integer
        enum:
          - 2
        description: Medium
      - type: integer
        enum:
          - 3
        description: Large
  workPerDate:
    type: string
    description: workPerDate
    verbs: []
  workPerDayList:
    type: array
    description: workPerDayList
    verbs: []
    items:
      type: string
    uniqueItems: true
  workRequired:
    type: integer
    description: Planned Hours
    verbs: []
  workRequiredExpression:
    type: string
    description: workRequiredExpression
    verbs: []
  workUnit:
    type: string
    description: workUnit
    verbs: []
    oneOf:
      - type: string
        enum:
          - M
        description: Minutes
      - type: string
        enum:
          - H
        description: Hours
      - type: string
        enum:
          - D
        description: Days
      - type: string
        enum:
          - W
        description: Weeks
      - type: string
        enum:
          - T
        description: Months
      - type: string
        enum:
          - EM
        description: Elapsed Minutes
      - type: string
        enum:
          - EH
        description: Elapsed Hours
      - type: string
        enum:
          - ED
        description: Elapsed Days
      - type: string
        enum:
          - EW
        description: Elapsed Weeks
      - type: string
        enum:
          - ET
        description: Elapsed Months
required:
  - name
  - projectID

--- USER METADATA ---

domain_knowledge: |
  When searching for a user by name, be sure to check firstName, lastName, and name fields.  
  Use the OR operator because the name may not appear in all three fields.  
  Searching by email or username may also be used.
fields:
  accessLevelID:
    type: string
    description: The ID of the Access Level associated with this user
    verbs:
      - GET
  accessLevelIDBackup:
    type: string
    description: accessLevelIDBackup
    verbs: []
  actualDeactivationDate:
    type: string
    description: actualDeactivationDate
    verbs: []
  address:
    type: string
    description: Address
    verbs: []
  address2:
    type: string
    description: Address2
    verbs: []
  alignAccessType:
    type: string
    description: alignAccessType
    verbs: []
  apiKeyCreateDate:
    type: string
    description: apiKeyCreateDate
    verbs: []
  avatarDate:
    type: string
    description: Profile Photo Date
    verbs: []
  avatarDownloadURL:
    type: string
    description: avatarDownloadURL
    verbs: []
  avatarSize:
    type: integer
    description: Profile Photo Size
    verbs: []
  avatarX:
    type: integer
    description: Profile Photo X
    verbs: []
  avatarY:
    type: integer
    description: Profile Photo Y
    verbs: []
  billingPerHour:
    type: number
    description: Billing Per Hour
    verbs: []
  categoryID:
    type: string
    description: Category ID
    verbs: []
  city:
    type: string
    description: City
    verbs: []
  companyID:
    type: string
    description: Company ID
    verbs: []
  costPerHour:
    type: number
    description: Cost Per Hour
    verbs: []
  country:
    type: string
    description: Country
    verbs: []
  customerID:
    type: string
    description: Customer ID
    verbs: []
  customFormsDescriptions:
    type: string
    description: customFormsDescriptions
    verbs: []
  customFormsGroupIDs:
    type: string
    description: customFormsGroupIDs
    verbs: []
  customFormsIDs:
    type: string
    description: customFormsIDs
    verbs:
      - GET
  customFormsNames:
    type: string
    description: customFormsNames
    verbs:
      - GET
  defaultHourTypeID:
    type: string
    description: Default Hour Type ID
    verbs: []
  defaultInterface:
    type: integer
    description: Default Interface
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Default
      - type: integer
        enum:
          - 1
        description: Project Management Home
      - type: integer
        enum:
          - 2
        description: TeamHome
      - type: integer
        enum:
          - 3
        description: Early Access
  delegationToID:
    type: string
    description: delegationToID
    verbs: []
  eauthUserID:
    type: string
    description: eauthUserID
    verbs: []
  emailAddr:
    type: string
    description: Email Address
    verbs:
      - GET
  enteredByID:
    type: string
    description: Entered By ID
    verbs:
      - GET
  entryDate:
    type: string
    description: Entry Date
    verbs:
      - GET
  extRefID:
    type: string
    description: External Reference ID
    verbs: []
  firstName:
    type: string
    description: First Name
    verbs:
      - GET
  fte:
    type: number
    description: FTE
    verbs: []
  hasAPIAccess:
    type: boolean
    description: Has API Access
    verbs: []
  hasDocuments:
    type: boolean
    description: Has Documents
    verbs: []
  hasNotes:
    type: boolean
    description: Has Notes
    verbs: []
  hasPassword:
    type: boolean
    description: hasPassword
    verbs: []
  hasProofLicense:
    type: boolean
    description: Has Proof License
    verbs: []
  hasQuicksilver:
    type: boolean
    description: hasQuicksilver
    verbs: []
  hasReservedTimes:
    type: boolean
    description: Has Time Off
    verbs: []
  homeGroupID:
    type: string
    description: The home group ID for this user
    verbs:
      - GET
  homeTeamID:
    type: string
    description: The home team ID for this user
    verbs:
      - GET
  ID:
    type: string
    description: User ID
    verbs:
      - GET
  isActive:
    type: boolean
    description: Is Active
    verbs: []
  isAdmin:
    type: boolean
    description: Is Admin
    verbs: []
  isBoxAuthenticated:
    type: boolean
    description: isBoxAuthenticated
    verbs: []
  isDropBoxAuthenticated:
    type: boolean
    description: isDropBoxAuthenticated
    verbs: []
  isGoogleAuthenticated:
    type: boolean
    description: isGoogleAuthenticated
    verbs: []
  isLimitedOrFullAdmin:
    type: boolean
    description: isLimitedOrFullAdmin
    verbs: []
  isSharePointAuthenticated:
    type: boolean
    description: isSharePointAuthenticated
    verbs: []
  isWebDAMAuthenticated:
    type: boolean
    description: isWebDAMAuthenticated
    verbs: []
  isWorkfrontDAMAuthenticated:
    type: boolean
    description: isWorkfrontDAMAuthenticated
    verbs: []
  lastEnteredNoteID:
    type: string
    description: Last Entered Note ID
    verbs: []
  lastLoginDate:
    type: string
    description: Last Login Date
    verbs:
      - GET
  lastName:
    type: string
    description: Last Name
    verbs:
      - GET
  lastNoteID:
    type: string
    description: Last Note ID
    verbs: []
  lastProofApprovalSync:
    type: string
    description: lastProofApprovalSync
    verbs: []
  lastUpdateDate:
    type: string
    description: Last Update Date
    verbs:
      - GET
  lastUpdatedByID:
    type: string
    description: Last Updated By ID
    verbs:
      - GET
  latestUpdateNoteID:
    type: string
    description: Latest Update Note ID
    verbs: []
  layoutTemplateID:
    type: string
    description: Layout Template ID
    verbs:
      - GET
  licenseType:
    type: string
    description: License type associated with this user
    verbs:
      - GET
    oneOf:
      - type: string
        enum:
          - F
        description: Standard
      - type: string
        enum:
          - C
        description: Light
      - type: string
        enum:
          - R
        description: Contributor
      - type: string
        enum:
          - X
        description: External
  licenseUpgradedDate:
    type: string
    description: Date Issued
    verbs: []
  locale:
    type: string
    description: Locale
    verbs: []
  loginBlocked:
    type: boolean
    description: Is Login Blocked
    verbs: []
  loginCount:
    type: integer
    description: Login Count
    verbs: []
  logTimeInDays:
    type: boolean
    description: logTimeInDays
    verbs: []
  managerID:
    type: string
    description: Manager ID
    verbs:
      - GET
  mobilePhoneNumber:
    type: string
    description: Mobile Phone
    verbs: []
  myInfo:
    type: string
    description: Talk to Me About
    verbs: []
  name:
    type: string
    description: Name
    verbs:
      - GET
  parentGroupIDs:
    type: array
    description: parentGroupIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  password:
    type: string
    description: Password
    verbs: []
  passwordDate:
    type: string
    description: Password Date
    verbs: []
  persona:
    type: string
    description: Persona
    verbs: []
  phoneExtension:
    type: string
    description: Extension
    verbs: []
  phoneNumber:
    type: string
    description: Phone number
    verbs: []
  portalProfileID:
    type: string
    description: Portal Profile ID
    verbs: []
  postalCode:
    type: string
    description: Postal Code
    verbs: []
  proofAccountRole:
    type: integer
    description: proofAccountRole
    verbs: []
  proofHQUsername:
    type: string
    description: proofHQUsername
    verbs: []
  proofsToApprove:
    type: array
    description: proofsToApprove
    verbs: []
    items:
      type: string
    uniqueItems: true
  receiveEmails:
    type: boolean
    description: receiveEmails
    verbs: []
  registrationExpireDate:
    type: string
    description: Registration Expiration Date
    verbs: []
  resetPasswordExpireDate:
    type: string
    description: Reset Password Expire Date
    verbs: []
  resourcePoolID:
    type: string
    description: Legacy Resource Pool ID
    verbs: []
  roleID:
    type: string
    description: Role ID
    verbs:
      - GET
  scheduledDeactivationDate:
    type: string
    description: Scheduled Deactivation Date
    verbs: []
  scheduleID:
    type: string
    description: Schedule ID
    verbs: []
  ssoAccessOnly:
    type: boolean
    description: SSO Access Only
    verbs: []
  ssoUsername:
    type: string
    description: SSO Username
    verbs: []
  state:
    type: string
    description: State
    verbs: []
  statusUpdate:
    type: string
    description: Status Update
    verbs: []
  teamsListString:
    type: string
    description: teamsListString
    verbs: []
  timesheetProfileID:
    type: string
    description: Timesheet Profile ID
    verbs: []
  timeZone:
    type: string
    description: Time Zone
    verbs: []
  title:
    type: string
    description: Title
    verbs:
      - GET
  uiTemplateID:
    type: string
    description: Layout Template ID
    verbs: []
  username:
    type: string
    description: Username
    verbs:
      - GET
  uumUserID:
    type: string
    description: uumUserID
    verbs: []
  webDAVProfile:
    type: array
    description: Web DAV Profile
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ALL
          description: All Projects
        - type: string
          enum:
            - CUR
          description: Current Projects
        - type: string
          enum:
            - FAV
          description: Favorite Projects
    uniqueItems: true
  workHoursPerDay:
    type: number
    description: workHoursPerDay
    verbs: []
  workTime:
    type: number
    description: Work Time
    verbs: []

User: {question}
Assistant: I'll help you with defining the correct call with the correct ObjCode, Fields, and Filters.

<thinking>
I need to understand the user's request and determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy based on the metadata provided).
</thinking>

<answer>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront’s custom object names and metadata to do the same using the context given below.

Your role is to interpret a user’s natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy). Your output should follow this structure:

{{
  "prompt": "What are all the tasks with high priority due next week?",
  "expected_response": {{
    "tool": "fetch_data",
    "objCode": "TASK",
    "fields": ["ID", "name", "priority", "plannedCompletionDate"],
    "filters": {{
      "priority": "High",
      "plannedCompletionDate": "$$TODAYbw+7d"
    }}
  }}
}}

 Workfront Object Context You Can Use

 Core Objects:
TASK: Represents individual tasks.
PROJ: Represents projects.
USER: Represents users (people in the org).

--- PROJECT METADATA ---
domain_knowledge:|
  Projects represent the main container of work that is to be completed. A Project may contain many different tasks or issues. Projects have both a plannedCompletionDate 
  which represents an estimate or planned completion date, and also a actualCompletionDate which represents the actual completion date of the project. If you are looking
  for when things actually completed use the actualCompletionDate, if you are wanting to know when something is planned to be completed use the plannedCompletionDate.
  Default Statuses include: Current, Dead, On Hold, Planning, Complete, Requested, Approved, Rejected, and Idea
fields:
  accessorIDs:
    type: array
    description: accessorIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  actualBenefit:
    type: number
    description: Actual Benefit
    verbs: []
  actualBillableExpenseCost:
    type: number
    description: actualBillableExpenseCost
    verbs: []
  actualCompletionDate:
    type: string
    description: The date in which this project was actually completed - this is not the same as the planned completion date which was more of an estimate
    verbs:
      - GET
      - PUT
      - POST
  actualCost:
    type: number
    description: The actual cost of this Project
    verbs:
      - GET
  actualDurationExpression:
    type: string
    description: actualDurationExpression
    verbs: []
  actualDurationMinutes:
    type: integer
    description: Actual Duration Minutes
    verbs: []
  actualExpenseCost:
    type: number
    description: Actual Expense Cost
    verbs: []
  actualHoursLastMonth:
    type: number
    description: Actual Hours Last Month
    verbs: []
  actualHoursLastThreeMonths:
    type: number
    description: Actual Hours Last Three Months
    verbs: []
  actualHoursThisMonth:
    type: number
    description: Actual Hours This Month
    verbs: []
  actualHoursTwoMonthsAgo:
    type: number
    description: Actual Hours Two Months Ago
    verbs: []
  actualLaborCost:
    type: number
    description: Actual Labor Cost
    verbs: []
  actualLaborRevenue:
    type: number
    description: Actual Labor Revenue
    verbs: []
  actualNonBillableExpenseCost:
    type: number
    description: actualNonBillableExpenseCost
    verbs: []
  actualNonLaborCost:
    type: number
    description: Actual Non-Labor Cost
    verbs: []
  actualNonLaborRevenue:
    type: number
    description: Actual Non-Labor Revenue
    verbs: []
  actualRevenue:
    type: number
    description: Actual Revenue
    verbs: []
  actualRiskCost:
    type: number
    description: Actual Risk Cost
    verbs: []
  actualStartDate:
    type: string
    description: The official start date of this project
    verbs:
      - GET
      - PUT
      - POST
  actualValue:
    type: number
    description: Actual Value
    verbs: []
  actualWorkRequired:
    type: integer
    description: Actual Hours
    verbs: []
  actualWorkRequiredDouble:
    type: number
    description: actualWorkRequiredDouble
    verbs: []
  actualWorkRequiredExpression:
    type: string
    description: actualWorkRequiredExpression
    verbs: []
  alignment:
    type: number
    description: Alignment
    verbs: []
  alignmentScoreCardID:
    type: string
    description: Alignment Scorecard ID
    verbs: []
  allApprovedHours:
    type: number
    description: allApprovedHours
    verbs: []
  allUnapprovedHours:
    type: number
    description: allUnapprovedHours
    verbs: []
  approvalCompletionDate:
    type: string
    description: Approval Path Completion Date
    verbs: []
  approvalEstStartDate:
    type: string
    description: Approval Estimated Start Date
    verbs: []
  approvalPlannedStartDate:
    type: string
    description: Approval Planned Start Date
    verbs: []
  approvalPlannedStartDay:
    type: number
    description: approvalPlannedStartDay
    verbs: []
  approvalProcessID:
    type: string
    description: Approval Processes ID
    verbs: []
  approvalProjectedStartDate:
    type: string
    description: Approval Projected Start Date
    verbs: []
  approvalStartDate:
    type: string
    description: Approval Path Start Date
    verbs: []
  approversString:
    type: string
    description: approversString
    verbs: []
  attachedRateCardID:
    type: string
    description: attachedRateCardID
    verbs: []
  attachedRateCardName:
    type: string
    description: attachedRateCardName
    verbs: []
  auditTypes:
    type: array
    description: Audit Types
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ST
          description: Status Change
        - type: string
          enum:
            - AA
          description: Attachment Action
        - type: string
          enum:
            - SC
          description: Scope Change
        - type: string
          enum:
            - "NO"
          description: Note
        - type: string
          enum:
            - GE
          description: General Edit
        - type: string
          enum:
            - CM
          description: Combined Entry
        - type: string
          enum:
            - ER
          description: Error Entry
        - type: string
          enum:
            - SB
          description: Subscription Change
    uniqueItems: true
  autoBaselineRecurOn:
    type: integer
    description: Auto Baseline Recur On
    verbs: []
  autoBaselineRecurrenceType:
    type: string
    description: Auto Baseline Recurrence Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - D
        description: Daily
      - type: string
        enum:
          - W
        description: Weekly
      - type: string
        enum:
          - BW
        description: Bi-weekly
      - type: string
        enum:
          - SM
        description: Semi-monthly
      - type: string
        enum:
          - M
        description: Monthly
      - type: string
        enum:
          - C
        description: Custom
  BCCompletionState:
    type: array
    description: BC Completion State
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - SUMMARY
          description: Summary
        - type: string
          enum:
            - INFO
          description: Project Info
        - type: string
          enum:
            - DEL
          description: Goals
        - type: string
          enum:
            - RISK
          description: Risks
        - type: string
          enum:
            - COST
          description: Costs
        - type: string
          enum:
            - ALIGN
          description: Alignment
        - type: string
          enum:
            - DE
          description: Custom Data
    uniqueItems: true
  bcwp:
    type: number
    description: BCWP
    verbs: []
  bcws:
    type: number
    description: BCWS
    verbs: []
  billedRevenue:
    type: number
    description: Billed Revenue
    verbs: []
  budget:
    type: number
    description: The official budget associated with this Project
    verbs:
      - GET
      - PUT
      - POST
  budgetedCompletionDate:
    type: string
    description: Budgeted Completion Date
    verbs: []
  budgetedCost:
    type: number
    description: Budgeted Cost
    verbs: []
  budgetedHours:
    type: number
    description: Budgeted Hours
    verbs: []
  budgetedLaborCost:
    type: number
    description: Budgeted Labor Cost
    verbs: []
  budgetedLaborCostSource:
    type: string
    description: budgetedLaborCostSource
    verbs: []
  budgetedStartDate:
    type: string
    description: Budgeted Start Date
    verbs: []
  budgetStatus:
    type: integer
    description: Budget Status
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Not Included
      - type: integer
        enum:
          - 1
        description: Included but not Calculated
      - type: integer
        enum:
          - 2
        description: Included and Calculated
  businessCaseStatusLabel:
    type: string
    description: Business Case Status
    verbs: []
  categoryID:
    type: string
    description: Category ID
    verbs: []
  companyID:
    type: string
    description: Company ID
    verbs: []
  companyRatesOverride:
    type: boolean
    description: Company Rates Override
    verbs: []
  completedHours:
    type: number
    description: Completed Hours
    verbs: []
  completedStoryPoints:
    type: number
    description: Completed Story Points
    verbs: []
  completionType:
    type: string
    description: Completion Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - MAN
        description: Manual
      - type: string
        enum:
          - AUT
        description: Automatic
  condition:
    type: string
    description: Condition
    verbs: []
    oneOf:
      - type: string
        enum:
          - "ON"
        description: On Target
      - type: string
        enum:
          - AR
        description: At Risk
      - type: string
        enum:
          - IT
        description: In Trouble
  conditionType:
    type: string
    description: Condition Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - PG
        description: Progress Status
      - type: string
        enum:
          - MN
        description: Manual
  convertedOpTaskEntryDate:
    type: string
    description: Converted Issue Entry Date
    verbs: []
  convertedOpTaskID:
    type: string
    description: convertedOpTaskID
    verbs: []
  convertedOpTaskName:
    type: string
    description: Converted Issue Name
    verbs: []
  convertedOpTaskOriginatorID:
    type: string
    description: convertedOpTaskOriginatorID
    verbs: []
  cpi:
    type: number
    description: Cost Performance Index
    verbs: []
  csi:
    type: number
    description: Cost Schedule Performance Index
    verbs: []
  currency:
    type: string
    description: Currency
    verbs: []
  currentApprovalStepID:
    type: string
    description: Current Approval Stage ID
    verbs: []
  customerID:
    type: string
    description: Customer ID
    verbs: []
  customFormsDescriptions:
    type: string
    description: customFormsDescriptions
    verbs: []
  customFormsGroupIDs:
    type: string
    description: customFormsGroupIDs
    verbs: []
  customFormsIDs:
    type: string
    description: The id of the custom forms attached to this Project
    verbs:
      - GET
  customFormsNames:
    type: string
    description: The name of the custom forms attached to this Project
    verbs:
      - GET
  defaultForbiddenContributeActions:
    type: array
    description: defaultForbiddenContributeActions
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ADD
          description: Add
        - type: string
          enum:
            - EDIT
          description: Edit
        - type: string
          enum:
            - LIMITED_EDIT
          description: Limited Edit
        - type: string
          enum:
            - VIEW
          description: View
        - type: string
          enum:
            - DELETE
          description: Delete
        - type: string
          enum:
            - LEGACY_APPROVE
          description: Approve
        - type: string
          enum:
            - LEGACY_REQUEST_PROJECT
          description: Request
        - type: string
          enum:
            - CHANGE_STATUS
          description: Change Status
        - type: string
          enum:
            - EDIT_CUSTOMDATA
          description: Edit Custom Form
        - type: string
          enum:
            - EDIT_APPROVALPROCESS
          description: Edit Approval Process
        - type: string
          enum:
            - EDIT_ROLE_GROUP
          description: Change a User's Role or Group
        - type: string
          enum:
            - EDIT_ACCESSLEVEL
          description: Change a User's Access Level
        - type: string
          enum:
            - ADD_TASKS
          description: Add Tasks
        - type: string
          enum:
            - ADD_ISSUES
          description: Add Issues
        - type: string
          enum:
            - EDIT_ASSIGNMENTS
          description: Make Assignments
        - type: string
          enum:
            - ADD_DOCUMENTS
          description: Add Documents
        - type: string
          enum:
            - EDIT_FINANCE
          description: Manage Finance
        - type: string
          enum:
            - VIEW_FINANCE
          description: View Finance
        - type: string
          enum:
            - ADD_HOURS
          description: Log Hours
        - type: string
          enum:
            - ADD_EXPENSES
          description: Add Expenses
        - type: string
          enum:
            - SHARE
          description: Share
        - type: string
          enum:
            - PROOF
          description: Proof
        - type: string
          enum:
            - DOWNLOAD
          description: Download
        - type: string
          enum:
            - SHARE_SYSTEMWIDE
          description: Share System-wide
        - type: string
          enum:
            - VIEW_CONTACTINFO
          description: View Contact Info
        - type: string
          enum:
            - ATTACH_CUSTOMFORM
          description: Attach to Custom Data
        - type: string
          enum:
            - BUDGETING_INFORMATION
          description: Edit priorities and budget hours in the Planner
        - type: string
          enum:
            - PLANNED_HOURS_CONTOURING
          description: Update Planned Hours in the Workload Balancer
        - type: string
          enum:
            - ADD_TO_CUSTOM_FORMS
          description: Add to Custom Forms
        - type: string
          enum:
            - ADD_BOOKINGS
          description: Add Bookings
        - type: string
          enum:
            - VIEW_COST_RATES
          description: View Cost Rates
        - type: string
          enum:
            - VIEW_BILLING_RATES
          description: View Billing Rates
        - type: string
          enum:
            - VIEW_GENERAL_FINANCE
          description: View general finance
        - type: string
          enum:
            - EDIT_COST_RATES
          description: Manage Cost Rates
        - type: string
          enum:
            - EDIT_BILLING_RATES
          description: Manage Billing Rates
        - type: string
          enum:
            - EDIT_GENERAL_FINANCE
          description: Manage general finance
        - type: string
          enum:
            - EDIT_TEAMS_I_AM_ON
          description: Edit teams I am on
        - type: string
          enum:
            - EDIT_TEAMS_I_GROUP_ADMIN
          description: Edit teams in groups I manage (Group Admins only)
        - type: string
          enum:
            - EDIT_SYSTEMWIDE
          description: enum.actiontypeenum.editystemwide
    uniqueItems: true
  defaultForbiddenManageActions:
    type: array
    description: defaultForbiddenManageActions
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ADD
          description: Add
        - type: string
          enum:
            - EDIT
          description: Edit
        - type: string
          enum:
            - LIMITED_EDIT
          description: Limited Edit
        - type: string
          enum:
            - VIEW
          description: View
        - type: string
          enum:
            - DELETE
          description: Delete
        - type: string
          enum:
            - LEGACY_APPROVE
          description: Approve
        - type: string
          enum:
            - LEGACY_REQUEST_PROJECT
          description: Request
        - type: string
          enum:
            - CHANGE_STATUS
          description: Change Status
        - type: string
          enum:
            - EDIT_CUSTOMDATA
          description: Edit Custom Form
        - type: string
          enum:
            - EDIT_APPROVALPROCESS
          description: Edit Approval Process
        - type: string
          enum:
            - EDIT_ROLE_GROUP
          description: Change a User's Role or Group
        - type: string
          enum:
            - EDIT_ACCESSLEVEL
          description: Change a User's Access Level
        - type: string
          enum:
            - ADD_TASKS
          description: Add Tasks
        - type: string
          enum:
            - ADD_ISSUES
          description: Add Issues
        - type: string
          enum:
            - EDIT_ASSIGNMENTS
          description: Make Assignments
        - type: string
          enum:
            - ADD_DOCUMENTS
          description: Add Documents
        - type: string
          enum:
            - EDIT_FINANCE
          description: Manage Finance
        - type: string
          enum:
            - VIEW_FINANCE
          description: View Finance
        - type: string
          enum:
            - ADD_HOURS
          description: Log Hours
        - type: string
          enum:
            - ADD_EXPENSES
          description: Add Expenses
        - type: string
          enum:
            - SHARE
          description: Share
        - type: string
          enum:
            - PROOF
          description: Proof
        - type: string
          enum:
            - DOWNLOAD
          description: Download
        - type: string
          enum:
            - SHARE_SYSTEMWIDE
          description: Share System-wide
        - type: string
          enum:
            - VIEW_CONTACTINFO
          description: View Contact Info
        - type: string
          enum:
            - ATTACH_CUSTOMFORM
          description: Attach to Custom Data
        - type: string
          enum:
            - BUDGETING_INFORMATION
          description: Edit priorities and budget hours in the Planner
        - type: string
          enum:
            - PLANNED_HOURS_CONTOURING
          description: Update Planned Hours in the Workload Balancer
        - type: string
          enum:
            - ADD_TO_CUSTOM_FORMS
          description: Add to Custom Forms
        - type: string
          enum:
            - ADD_BOOKINGS
          description: Add Bookings
        - type: string
          enum:
            - VIEW_COST_RATES
          description: View Cost Rates
        - type: string
          enum:
            - VIEW_BILLING_RATES
          description: View Billing Rates
        - type: string
          enum:
            - VIEW_GENERAL_FINANCE
          description: View general finance
        - type: string
          enum:
            - EDIT_COST_RATES
          description: Manage Cost Rates
        - type: string
          enum:
            - EDIT_BILLING_RATES
          description: Manage Billing Rates
        - type: string
          enum:
            - EDIT_GENERAL_FINANCE
          description: Manage general finance
        - type: string
          enum:
            - EDIT_TEAMS_I_AM_ON
          description: Edit teams I am on
        - type: string
          enum:
            - EDIT_TEAMS_I_GROUP_ADMIN
          description: Edit teams in groups I manage (Group Admins only)
        - type: string
          enum:
            - EDIT_SYSTEMWIDE
          description: enum.actiontypeenum.editystemwide
    uniqueItems: true
  defaultForbiddenViewActions:
    type: array
    description: defaultForbiddenViewActions
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ADD
          description: Add
        - type: string
          enum:
            - EDIT
          description: Edit
        - type: string
          enum:
            - LIMITED_EDIT
          description: Limited Edit
        - type: string
          enum:
            - VIEW
          description: View
        - type: string
          enum:
            - DELETE
          description: Delete
        - type: string
          enum:
            - LEGACY_APPROVE
          description: Approve
        - type: string
          enum:
            - LEGACY_REQUEST_PROJECT
          description: Request
        - type: string
          enum:
            - CHANGE_STATUS
          description: Change Status
        - type: string
          enum:
            - EDIT_CUSTOMDATA
          description: Edit Custom Form
        - type: string
          enum:
            - EDIT_APPROVALPROCESS
          description: Edit Approval Process
        - type: string
          enum:
            - EDIT_ROLE_GROUP
          description: Change a User's Role or Group
        - type: string
          enum:
            - EDIT_ACCESSLEVEL
          description: Change a User's Access Level
        - type: string
          enum:
            - ADD_TASKS
          description: Add Tasks
        - type: string
          enum:
            - ADD_ISSUES
          description: Add Issues
        - type: string
          enum:
            - EDIT_ASSIGNMENTS
          description: Make Assignments
        - type: string
          enum:
            - ADD_DOCUMENTS
          description: Add Documents
        - type: string
          enum:
            - EDIT_FINANCE
          description: Manage Finance
        - type: string
          enum:
            - VIEW_FINANCE
          description: View Finance
        - type: string
          enum:
            - ADD_HOURS
          description: Log Hours
        - type: string
          enum:
            - ADD_EXPENSES
          description: Add Expenses
        - type: string
          enum:
            - SHARE
          description: Share
        - type: string
          enum:
            - PROOF
          description: Proof
        - type: string
          enum:
            - DOWNLOAD
          description: Download
        - type: string
          enum:
            - SHARE_SYSTEMWIDE
          description: Share System-wide
        - type: string
          enum:
            - VIEW_CONTACTINFO
          description: View Contact Info
        - type: string
          enum:
            - ATTACH_CUSTOMFORM
          description: Attach to Custom Data
        - type: string
          enum:
            - BUDGETING_INFORMATION
          description: Edit priorities and budget hours in the Planner
        - type: string
          enum:
            - PLANNED_HOURS_CONTOURING
          description: Update Planned Hours in the Workload Balancer
        - type: string
          enum:
            - ADD_TO_CUSTOM_FORMS
          description: Add to Custom Forms
        - type: string
          enum:
            - ADD_BOOKINGS
          description: Add Bookings
        - type: string
          enum:
            - VIEW_COST_RATES
          description: View Cost Rates
        - type: string
          enum:
            - VIEW_BILLING_RATES
          description: View Billing Rates
        - type: string
          enum:
            - VIEW_GENERAL_FINANCE
          description: View general finance
        - type: string
          enum:
            - EDIT_COST_RATES
          description: Manage Cost Rates
        - type: string
          enum:
            - EDIT_BILLING_RATES
          description: Manage Billing Rates
        - type: string
          enum:
            - EDIT_GENERAL_FINANCE
          description: Manage general finance
        - type: string
          enum:
            - EDIT_TEAMS_I_AM_ON
          description: Edit teams I am on
        - type: string
          enum:
            - EDIT_TEAMS_I_GROUP_ADMIN
          description: Edit teams in groups I manage (Group Admins only)
        - type: string
          enum:
            - EDIT_SYSTEMWIDE
          description: enum.actiontypeenum.editystemwide
    uniqueItems: true
  deliverableScoreCardID:
    type: string
    description: Deliverable Scorecard ID
    verbs: []
  deliverableSuccessScore:
    type: number
    description: Goal Success Score
    verbs: []
  deliverableSuccessScoreRatio:
    type: number
    description: deliverableSuccessScoreRatio
    verbs: []
  description:
    type: string
    description: The detailed description of this Project
    verbs:
      - GET
      - PUT
      - POST
  displayOrder:
    type: integer
    description: Display Order
    verbs: []
  durationExpression:
    type: string
    description: durationExpression
    verbs: []
  durationMinutes:
    type: integer
    description: Duration in Minutes
    verbs: []
  eac:
    type: number
    description: EAC
    verbs: []
  eacCalculationMethod:
    type: string
    description: EAC Calculation Method
    verbs: []
    oneOf:
      - type: string
        enum:
          - P
        description: Calculate at project level
      - type: string
        enum:
          - T
        description: Roll up from tasks/subtasks
  enableAutoBaselines:
    type: boolean
    description: Enable Auto Baselines
    verbs: []
  enteredByID:
    type: string
    description: The ID of the user who entered this Project
    verbs:
      - GET
  entryDate:
    type: string
    description: The date in which this project was created
    verbs:
      - GET
  estCompletionDate:
    type: string
    description: The date in which this project is estimated to be complete. This is a calculated field.
    verbs:
      - GET
  estStartDate:
    type: string
    description: The data in which we expect to start working on this project
    verbs:
      - GET
  extRefID:
    type: string
    description: External Reference ID
    verbs: []
  filterHourTypes:
    type: boolean
    description: Filter Hour Types
    verbs: []
  financeLastUpdateDate:
    type: string
    description: Last Finance Update Date
    verbs: []
  fixedCost:
    type: number
    description: Fixed Cost
    verbs: []
  fixedEndDate:
    type: string
    description: Fixed End Date
    verbs: []
  fixedRevenue:
    type: number
    description: Fixed Revenue
    verbs: []
  fixedStartDate:
    type: string
    description: Fixed Start Date
    verbs: []
  goalHierarchy:
    type: string
    description: goalHierarchy
    verbs: []
  goalsString:
    type: string
    description: goalsString
    verbs: []
  groupID:
    type: string
    description: The ID of the group associated with this Project. Users can belong to Groups.
    verbs:
      - GET
      - PUT
      - POST
  hasBookings:
    type: boolean
    description: hasBookings
    verbs: []
  hasBudgetConflict:
    type: boolean
    description: Has Budget Conflict
    verbs: []
  hasCalcError:
    type: boolean
    description: Has Calc Error
    verbs: []
  hasCompletionConstraint:
    type: boolean
    description: hasCompletionConstraint
    verbs: []
  hasDocuments:
    type: boolean
    description: Has Documents
    verbs: []
  hasExpenses:
    type: boolean
    description: Has Expenses
    verbs: []
  hasMessages:
    type: boolean
    description: Has Messages
    verbs: []
  hasNotes:
    type: boolean
    description: Has Notes
    verbs: []
  hasRateOverride:
    type: boolean
    description: Has Billing Rate Override
    verbs: []
  hasResolvables:
    type: boolean
    description: Has Resolvables
    verbs: []
  hasStartConstraint:
    type: boolean
    description: hasStartConstraint
    verbs: []
  hasTimedNotifications:
    type: boolean
    description: Has Reminder Notifications
    verbs: []
  hasTimelineException:
    type: boolean
    description: hasTimelineException
    verbs: []
  ID:
    type: string
    description: The unique ID or identifier of this project
    verbs:
      - GET
      - PUT
      - POST
      - DELETE
  isOriginalPlannedHoursSet:
    type: boolean
    description: isOriginalPlannedHoursSet
    verbs: []
  isProjectDead:
    type: boolean
    description: isProjectDead
    verbs: []
  isStatusComplete:
    type: boolean
    description: isStatusComplete
    verbs: []
  lastCalcDate:
    type: string
    description: Last Calc Date
    verbs: []
  lastConditionNoteID:
    type: string
    description: Last Condition Note ID
    verbs: []
  lastNoteID:
    type: string
    description: Last Note ID
    verbs: []
  lastUpdateDate:
    type: string
    description: The date of the last update to this Project
    verbs:
      - GET
  lastUpdatedByID:
    type: string
    description: The ID of the user who last updated this Project
    verbs:
      - GET
  levelingMode:
    type: string
    description: Leveling Mode
    verbs: []
    oneOf:
      - type: string
        enum:
          - AUT
        description: Automatic
      - type: string
        enum:
          - MAN
        description: Manual
  lucidMigrationDate:
    type: string
    description: lucidMigrationDate
    verbs: []
  milestonePathID:
    type: string
    description: Milestone Path ID
    verbs: []
  name:
    type: string
    description: The name of this Project
    verbs:
      - GET
      - PUT
      - POST
  nextAutoBaselineDate:
    type: string
    description: Next Date
    verbs: []
  numberOfLinkedGoals:
    type: integer
    description: Number of linked Goals
    verbs: []
  numberOfOpenRisks:
    type: integer
    description: Number of Open Risks
    verbs: []
  numberOpenOpTasks:
    type: integer
    description: Number of Open Issues
    verbs: []
  olv:
    type: integer
    description: Optimization Log Value
    verbs: []
  optimizationScore:
    type: integer
    description: Optimization Score
    verbs: []
  originalWorkRequired:
    type: integer
    description: Original Planned Hours
    verbs: []
  ownerID:
    type: string
    description: The person who owns or is responsible for this Project (The Project Owner)
    verbs:
      - GET
      - PUT
      - POST
  ownerPrivileges:
    type: array
    description: Owner Privileges
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - APT
          description: Require time to be approved for this project
    uniqueItems: true
  pendingCalculation:
    type: string
    description: pendingCalculation
    verbs: []
  pendingUpdateMethods:
    type: array
    description: pendingUpdateMethods
    verbs: []
    items:
      type: string
    uniqueItems: true
  percentComplete:
    type: number
    description: The current percent complete of this Project
    verbs:
      - GET
  performanceIndexMethod:
    type: string
    description: Performance Index Method
    verbs: []
    oneOf:
      - type: string
        enum:
          - H
        description: Hour-Based
      - type: string
        enum:
          - C
        description: Cost-Based
  personal:
    type: boolean
    description: Personal
    verbs: []
  plannedBenefit:
    type: number
    description: Planned Benefit
    verbs: []
  plannedBillableExpenseCost:
    type: number
    description: plannedBillableExpenseCost
    verbs: []
  plannedCompletionDate:
    type: string
    description: The date in which we planned this project to complete - this is not the actual completion date
    verbs:
      - GET
  plannedCost:
    type: number
    description: Planned Cost
    verbs: []
  plannedDateAlignment:
    type: integer
    description: Planned Date Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Will be done after the planned completion date
      - type: integer
        enum:
          - 1
        description: Will be done on the planned completion date
      - type: integer
        enum:
          - 2
        description: Will be done before the planned completion date
      - type: integer
        enum:
          - 3
        description: No set planned date
  plannedExpenseCost:
    type: number
    description: Planned Expense Cost
    verbs: []
  plannedHoursAlignment:
    type: integer
    description: Planned Hours Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: More hours than planned
      - type: integer
        enum:
          - 1
        description: Same hours as planned
      - type: integer
        enum:
          - 2
        description: Less hours than planned
      - type: integer
        enum:
          - 3
        description: No planned hours set
  plannedLaborCost:
    type: number
    description: Planned Labor Cost
    verbs: []
  plannedLaborRevenue:
    type: number
    description: Planned Labor Revenue
    verbs: []
  plannedNonBillableExpenseCost:
    type: number
    description: plannedNonBillableExpenseCost
    verbs: []
  plannedNonLaborCost:
    type: number
    description: Planned Non-Labor Cost
    verbs: []
  plannedNonLaborRevenue:
    type: number
    description: Planned Non-Labor Revenue
    verbs: []
  plannedRevenue:
    type: number
    description: Planned Revenue
    verbs: []
  plannedRiskCost:
    type: number
    description: Planned Risk Cost
    verbs: []
  plannedStartDate:
    type: string
    description: The date in which we plan to start work on this Project
    verbs:
      - GET
      - PUT
      - POST
  plannedValue:
    type: number
    description: Planned Value
    verbs: []
  popAccountID:
    type: string
    description: Pop Account ID
    verbs: []
  portfolioID:
    type: string
    description: The ID of the portfolio this project belongs to
    verbs:
      - GET
      - PUT
      - POST
  portfolioPriority:
    type: integer
    description: Portfolio Priority
    verbs: []
  previousStatus:
    type: string
    description: Previous status
    verbs: []
    oneOf:
      - type: string
        enum:
          - CUR
        description: Current
      - type: string
        enum:
          - ONH
        description: On Hold
      - type: string
        enum:
          - PLN
        description: Planning
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - DED
        description: Dead
      - type: string
        enum:
          - REQ
        description: Requested
      - type: string
        enum:
          - APR
        description: Approved
      - type: string
        enum:
          - REJ
        description: Rejected
      - type: string
        enum:
          - IDA
        description: Idea
  previousStatusLabel:
    type: string
    description: previousStatusLabel
    verbs: []
  priority:
    type: integer
    description: "Priority.  Use the enum values, not the descriptions.  The only accepted values are: 0, 1, 2, 3, 4."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: integer
        enum:
          - 0
        description: None
      - type: integer
        enum:
          - 1
        description: Low
      - type: integer
        enum:
          - 2
        description: Normal
      - type: integer
        enum:
          - 3
        description: High
      - type: integer
        enum:
          - 4
        description: Urgent
  privateRateCardID:
    type: string
    description: privateRateCardID
    verbs: []
  programID:
    type: string
    description: The ID or the Program this project is associated with
    verbs:
      - GET
      - PUT
      - POST
  progressStatus:
    type: string
    description: Progress Status
    verbs: []
    oneOf:
      - type: string
        enum:
          - LT
        description: Late
      - type: string
        enum:
          - "ON"
        description: On Time
      - type: string
        enum:
          - BH
        description: Behind
      - type: string
        enum:
          - RK
        description: At Risk
  projectBudgetedCost:
    type: number
    description: This is a calculated field representing the total budgeted cost of this project
    verbs:
      - GET
  projectedCompletionDate:
    type: string
    description: Calculated field represented the projected completion date of this project
    verbs:
      - GET
  projectedStartDate:
    type: string
    description: Calculated field representing the projected start date of this project
    verbs:
      - GET
  projectNetValue:
    type: number
    description: Project Net Value
    verbs: []
  projectRoi:
    type: number
    description: Project ROI
    verbs: []
  queueDefID:
    type: string
    description: Queue Definition ID
    verbs: []
  referenceNumber:
    type: integer
    description: A system generated value used to reference this project
    verbs:
      - GET
  rejectionIssueID:
    type: string
    description: Rejection Issue ID
    verbs: []
  remainingCost:
    type: number
    description: Remaining Cost
    verbs: []
  remainingRevenue:
    type: number
    description: Remaining Revenue
    verbs: []
  remainingRiskCost:
    type: number
    description: Remaining Risk Cost
    verbs: []
  resourcePlannerBudgetedHours:
    type: number
    description: Resource Planner Budgeted Hours
    verbs: []
  resourcePlannerBudgetedLaborCost:
    type: number
    description: Resource Planner Budgeted Labor Cost
    verbs: []
  resourcePoolID:
    type: string
    description: Legacy Resource Pool ID
    verbs: []
  risk:
    type: integer
    description: Risk
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: None
      - type: integer
        enum:
          - 1
        description: Very Low
      - type: integer
        enum:
          - 2
        description: Low
      - type: integer
        enum:
          - 3
        description: Medium
      - type: integer
        enum:
          - 4
        description: High
      - type: integer
        enum:
          - 5
        description: Very High
  riskPerformanceIndex:
    type: number
    description: RPI
    verbs: []
  roi:
    type: number
    description: ROI
    verbs: []
  rootGroupID:
    type: string
    description: rootGroupID
    verbs: []
  scenarioPlannerBudgetedLaborCost:
    type: number
    description: scenarioPlannerBudgetedLaborCost
    verbs: []
  scheduleID:
    type: string
    description: Schedule ID
    verbs: []
  scheduleMode:
    type: string
    description: Schedule Mode
    verbs: []
    oneOf:
      - type: string
        enum:
          - S
        description: Start Date
      - type: string
        enum:
          - C
        description: Completion Date
  securityAncestorsDisabled:
    type: boolean
    description: securityAncestorsDisabled
    verbs: []
  selectedOnPortfolioOptimizer:
    type: boolean
    description: Selected on Portfolio Optimizer
    verbs: []
  showCondition:
    type: boolean
    description: showCondition
    verbs: []
  showStatus:
    type: boolean
    description: showStatus
    verbs: []
  spi:
    type: number
    description: Schedule Performance Index
    verbs: []
  sponsorID:
    type: string
    description: Sponsor ID
    verbs: []
  status:
    type: string
    description: "Status.  Use the enum values, not the descriptions.  The only accepted values are: CUR, ONH, PLN, CPL, DED, REQ, APR, REJ, IDA."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - CUR
        description: Current
      - type: string
        enum:
          - ONH
        description: On Hold
      - type: string
        enum:
          - PLN
        description: Planning
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - DED
        description: Dead
      - type: string
        enum:
          - REQ
        description: Requested
      - type: string
        enum:
          - APR
        description: Approved
      - type: string
        enum:
          - REJ
        description: Rejected
      - type: string
        enum:
          - IDA
        description: Idea
  statusLabel:
    type: string
    description: Status Label
    verbs: []
  statusUpdate:
    type: string
    description: Status Update
    verbs: []
  storiesWithoutPoints:
    type: integer
    description: Tasks Without Points
    verbs: []
  storyPointsPercentComplete:
    type: number
    description: Percent Complete (Story Points)
    verbs: []
  submittedByID:
    type: string
    description: The ID of the user who submitted this Project
    verbs:
      - GET
  subscribersListString:
    type: string
    description: subscribersListString
    verbs: []
  summaryCompletionType:
    type: string
    description: summaryCompletionType
    verbs: []
    oneOf:
      - type: string
        enum:
          - MAN
        description: Manual
      - type: string
        enum:
          - AUT
        description: Automatic
  syncBurndownDate:
    type: string
    description: syncBurndownDate
    verbs: []
  teamID:
    type: string
    description: The ID of the team associated with this Project
    verbs: []
  templateID:
    type: string
    description: The ID of the template that was used to create this project
    verbs:
      - GET
      - POST
  totalHours:
    type: number
    description: Total Hours
    verbs: []
  totalOpTaskCount:
    type: integer
    description: Total Issue Count
    verbs: []
  totalStoryPoints:
    type: number
    description: Total Story Points
    verbs: []
  totalTaskCount:
    type: integer
    description: Total Task Count
    verbs: []
  updateType:
    type: string
    description: Update Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - AUTO
        description: Automatic and On Change
      - type: string
        enum:
          - EVNT
        description: Change Only
      - type: string
        enum:
          - ATYO
        description: Automatic Only
      - type: string
        enum:
          - MANU
        description: Manual Only
  URL:
    type: string
    description: URL
    verbs: []
  useDateEffectiveExchangeRates:
    type: boolean
    description: useDateEffectiveExchangeRates
    verbs: []
  userTimeOff:
    type: string
    description: User Time Off
    verbs: []
    oneOf:
      - type: string
        enum:
          - CUTO
        description: Consider user time off in task durations
      - type: string
        enum:
          - IUTO
        description: Ignore user time off in task durations
  version:
    type: integer
    description: Version
    verbs: []
  work:
    type: number
    description: Planned Hours
    verbs: []
  workRequired:
    type: integer
    description: Planned Hours
    verbs: []
  workRequiredExpression:
    type: string
    description: workRequiredExpression
    verbs: []

--- TASK METADATA ---
domain_knowledge: |
  Tasks always belong to a project, so they always have the field projectID even when updating a task or making an assignment. 
  To update existing tasks, you must include the taskID in the request. 
  Tasks may also have subtasks, in which case they have a parent task (parentID) and a projectID.
  Tasks frequently have an assignment to a user, team, or role.
  Tasks can have both a plannedCompletionDate and a actualCompletionDate - if you want to know the actual date something was completed, use actualCompletionDate.
  Assignments can be added when creating an task by adding appropriate ids to the assignments array.
  A task name is always required.
  To set a Due Date, set the taskConstraint to MFO and the constraintDate to the desired date.  
  To set a Start Date, set the taskConstraint to MSO and the constraintDate to the desired date.
  Default statuses include: New, In Progress, and Complete
  Tasks include a taskNumber.  The task number shows the order the task appears in a list as shown to the user.  The user may referer to a task by its task number instead of its name.
  If a user refers to a task by its number insetead of its name, then it will need to be search by taskNumber instead of name to find the task id.  i.e. I want to rename task 3 to "New Task Name"
fields:
  accessorIDs:
    type: array
    description: accessorIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  actualBillableExpenseCost:
    type: number
    description: actualBillableExpenseCost
    verbs: []
  actualCompletionDate:
    type: string
    description: The date in which this task was actually completed
    verbs:
      - GET
      - PUT
      - POST
  actualCost:
    type: number
    description: Actual Cost
    verbs:
      - GET
      - PUT
      - POST
  actualDuration:
    type: number
    description: Actual Duration
    verbs: []
  actualDurationMinutes:
    type: integer
    description: Actual Duration Minutes
    verbs: []
  actualExpenseCost:
    type: number
    description: Actual Expense Cost
    verbs: []
  actualLaborCost:
    type: number
    description: Actual Labor Cost
    verbs: []
  actualLaborRevenue:
    type: number
    description: Actual Labor Revenue
    verbs: []
  actualNonBillableExpenseCost:
    type: number
    description: actualNonBillableExpenseCost
    verbs: []
  actualNonLaborCost:
    type: number
    description: Actual Non-Labor Cost
    verbs: []
  actualNonLaborRevenue:
    type: number
    description: Actual Non-Labor Revenue
    verbs: []
  actualRevenue:
    type: number
    description: Actual Revenue
    verbs: []
  actualStartDate:
    type: string
    description: Actual Start Date
    verbs:
      - GET
      - PUT
      - POST
  actualWork:
    type: number
    description: Actual Hours
    verbs: []
  actualWorkRequired:
    type: integer
    description: Actual Hours
    verbs: []
  actualWorkRequiredDouble:
    type: number
    description: actualWorkRequiredDouble
    verbs: []
  approvalCompletionDate:
    type: string
    description: Approval Path Completion Date
    verbs: []
  approvalEstStartDate:
    type: string
    description: Approval Estimated Start Date
    verbs: []
  approvalPlannedStartDate:
    type: string
    description: Approval Planned Start Date
    verbs: []
  approvalPlannedStartDay:
    type: number
    description: approvalPlannedStartDay
    verbs: []
  approvalProcessID:
    type: string
    description: Approval Processes ID
    verbs: []
  approvalProjectedStartDate:
    type: string
    description: Approval Projected Start Date
    verbs: []
  approvalStartDate:
    type: string
    description: Approval Path Start Date
    verbs: []
  approversString:
    type: string
    description: approversString
    verbs: []
  assignments:
    type: array
    description: User, Role, or Team ID  assigned to this object for use when more than one assignment is needed
    verbs:
      - POST
      - GET
      - PUT
    items:
      type: object
      properties:
        assignedToID:
          type: string
          description: User, Role, or Team ID  assigned to this object
          verbs:
            - GET
            - POST
            - PUT
  assignedToID:
    type: string
    description: Assigned To ID
    verbs:
      - POST
      - GET
      - PUT
  assignmentsListString:
    type: string
    description: assignmentsListString
    verbs: []
  auditNote:
    type: string
    description: auditNote
    verbs: []
  auditTypes:
    type: array
    description: Audit Types
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ST
          description: Status Change
        - type: string
          enum:
            - AA
          description: Attachment Action
        - type: string
          enum:
            - SC
          description: Scope Change
        - type: string
          enum:
            - "NO"
          description: Note
        - type: string
          enum:
            - GE
          description: General Edit
        - type: string
          enum:
            - CM
          description: Combined Entry
        - type: string
          enum:
            - ER
          description: Error Entry
        - type: string
          enum:
            - SB
          description: Subscription Change
    uniqueItems: true
  auditUserIDs:
    type: array
    description: auditUserIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  backlogOrder:
    type: integer
    description: Order
    verbs: []
  backlogParent:
    type: string
    description: Parent
    verbs: []
  bcwp:
    type: number
    description: BCWP
    verbs: []
  bcws:
    type: number
    description: BCWS
    verbs: []
  billingAmount:
    type: number
    description: Cap or Fixed Amount
    verbs: []
  billingRecordID:
    type: string
    description: Billing Record ID
    verbs: []
  canStart:
    type: boolean
    description: Can Start
    verbs:
      - GET
      - PUT
      - POST
  categoryID:
    type: string
    description: Category ID
    verbs: []
  childrenIds:
    type: array
    description: childrenIds
    verbs: []
    items:
      type: string
    uniqueItems: true
  color:
    type: string
    description: Color
    verbs: []
  commitDate:
    type: string
    description: Commit Date
    verbs: []
  commitDateRange:
    type: string
    description: commitDateRange
    verbs: []
  completionPendingDate:
    type: string
    description: Completion Pending Date
    verbs: []
  condition:
    type: integer
    description: "Condition.  Use the enum values, not the descriptions.  The only accepted values are: 0, 1, 2."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: integer
        enum:
          - 0
        description: Going Smoothly
      - type: integer
        enum:
          - 1
        description: Some Concerns
      - type: integer
        enum:
          - 2
        description: Major Roadblocks
  constraintDate:
    type: string
    description: Constraint Date
    verbs:
      - GET
      - PUT
      - POST
  convertedOpTaskEntryDate:
    type: string
    description: Converted Issue Entry Date
    verbs: []
  convertedOpTaskID:
    type: string
    description: convertedOpTaskID
    verbs: []
  convertedOpTaskName:
    type: string
    description: Converted Issue Name
    verbs: []
  convertedOpTaskOriginatorID:
    type: string
    description: convertedOpTaskOriginatorID
    verbs: []
  costAmount:
    type: number
    description: Cost Amount
    verbs: []
  costType:
    type: string
    description: Cost Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - NOC
        description: No Cost
      - type: string
        enum:
          - FHR
        description: Fixed Hourly
      - type: string
        enum:
          - UHR
        description: User Hourly
      - type: string
        enum:
          - RHC
        description: Role Hourly
  cpi:
    type: number
    description: Cost Performance Index
    verbs: []
  csi:
    type: number
    description: Cost Schedule Performance Index
    verbs: []
  currentApprovalStepID:
    type: string
    description: Current Approval Stage ID
    verbs: []
  customerID:
    type: string
    description: Customer ID
    verbs: []
  customFormsDescriptions:
    type: string
    description: customFormsDescriptions
    verbs: []
  customFormsGroupIDs:
    type: string
    description: customFormsGroupIDs
    verbs: []
  customFormsIDs:
    type: string
    description: customFormsIDs
    verbs:
      - GET
  customFormsNames:
    type: string
    description: customFormsNames
    verbs:
      - GET
  daysLate:
    type: integer
    description: Days Late
    verbs: []
  description:
    type: string
    description: Description
    verbs:
      - GET
      - PUT
      - POST
  duration:
    type: number
    description: Duration.  Use this field to set the duration of the task.  The duration is the amount of time it will take to complete the task.  Always set durationUnit when setting duration.
    verbs:
      - GET
      - PUT
      - POST
  durationExpression:
    type: string
    description: durationExpression
    verbs: []
  durationMinutes:
    type: integer
    description: Duration in Minutes
    verbs: []
  durationType:
    type: string
    description: Duration Type
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - W
        description: Calculated Work
      - type: string
        enum:
          - D
        description: Effort Driven
      - type: string
        enum:
          - A
        description: Calculated Assignment
      - type: string
        enum:
          - S
        description: Simple
  durationUnit:
    type: string
    description: Duration Unit
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - M
        description: Minutes
      - type: string
        enum:
          - H
        description: Hours
      - type: string
        enum:
          - D
        description: Days
      - type: string
        enum:
          - W
        description: Weeks
      - type: string
        enum:
          - T
        description: Months
      - type: string
        enum:
          - EM
        description: Elapsed Minutes
      - type: string
        enum:
          - EH
        description: Elapsed Hours
      - type: string
        enum:
          - ED
        description: Elapsed Days
      - type: string
        enum:
          - EW
        description: Elapsed Weeks
      - type: string
        enum:
          - ET
        description: Elapsed Months
  eac:
    type: number
    description: EAC
    verbs: []
  enteredByID:
    type: string
    description: Entered By ID
    verbs:
      - GET
  entryDate:
    type: string
    description: The date in which this task was created or entered
    verbs:
      - GET
  estCompletionDate:
    type: string
    description: Estimated Due Date
    verbs: []
  estimate:
    type: number
    description: Estimate
    verbs:
      - GET
  estStartDate:
    type: string
    description: Estimated Start Date
    verbs: []
  extRefID:
    type: string
    description: External Reference ID
    verbs: []
  groupID:
    type: string
    description: Group ID
    verbs:
      - GET
      - PUT
      - POST
  handoffDate:
    type: string
    description: Handoff Date
    verbs: []
  hasBookings:
    type: boolean
    description: hasBookings
    verbs: []
  hasCompletionConstraint:
    type: boolean
    description: hasCompletionConstraint
    verbs: []
  hasDocuments:
    type: boolean
    description: Has Documents
    verbs: []
  hasExpenses:
    type: boolean
    description: Has Expenses
    verbs: []
  hasMessages:
    type: boolean
    description: Has Messages
    verbs: []
  hasNotes:
    type: boolean
    description: Has Notes
    verbs: []
  hasResolvables:
    type: boolean
    description: Has Resolvables
    verbs: []
  hasStartConstraint:
    type: boolean
    description: hasStartConstraint
    verbs: []
  hasTimedNotifications:
    type: boolean
    description: Has Reminder Notifications
    verbs: []
  hoursPerPoint:
    type: number
    description: hoursPerPoint
    verbs: []
  ID:
    type: string
    description: Task ID
    verbs:
      - GET
      - PUT
      - POST
      - DELETE
  indent:
    type: integer
    description: Indent
    verbs: []
  isAgile:
    type: boolean
    description: Agile
    verbs: []
  isCritical:
    type: boolean
    description: Is Critical
    verbs: []
  isDurationLocked:
    type: boolean
    description: Duration Locked
    verbs: []
  isLevelingExcluded:
    type: boolean
    description: Exclude from Leveling
    verbs: []
  isReady:
    type: boolean
    description: Ready
    verbs: []
  isStatusComplete:
    type: boolean
    description: isStatusComplete
    verbs: []
  isWorkRequiredLocked:
    type: boolean
    description: Worked Locked
    verbs: []
  iterationID:
    type: string
    description: Iteration ID
    verbs:
      - GET
      - PUT
      - POST
  kanbanBoardID:
    type: string
    description: Kanban Board ID
    verbs: []
  kanbanFlag:
    type: string
    description: Kanban Flag
    verbs: []
    oneOf:
      - type: string
        enum:
          - ONTRACK
        description: On Track
      - type: string
        enum:
          - ISBLOCKED
        description: Is Blocked
      - type: string
        enum:
          - READYTOPULL
        description: Ready to Pull
  lastConditionNoteID:
    type: string
    description: Last Condition Note ID
    verbs: []
  lastNoteID:
    type: string
    description: Last Note ID
    verbs: []
  lastUpdateDate:
    type: string
    description: Last Update Date
    verbs:
      - GET
  lastUpdatedByID:
    type: string
    description: Last Updated By ID
    verbs:
      - GET
  levelingStartDelay:
    type: number
    description: Leveling Delay
    verbs: []
  levelingStartDelayExpression:
    type: string
    description: levelingStartDelayExpression
    verbs: []
  levelingStartDelayMinutes:
    type: integer
    description: levelingStartDelayMinutes
    verbs: []
  milestoneID:
    type: string
    description: Milestone ID
    verbs: []
  name:
    type: string
    description: Name
    verbs:
      - GET
      - PUT
      - POST
  numberOfChildren:
    type: integer
    description: Number of Children
    verbs: []
  numberOpenOpTasks:
    type: integer
    description: Number of Open Issues
    verbs: []
  olv:
    type: integer
    description: Optimization Log Value
    verbs: []
  originalActualWorkRequired:
    type: integer
    description: originalActualWorkRequired
    verbs: []
  originalActualWorkRequiredDouble:
    type: number
    description: originalActualWorkRequiredDouble
    verbs: []
  originalDuration:
    type: integer
    description: Original Duration
    verbs: []
  originalWorkRequired:
    type: integer
    description: Original Planned Hours
    verbs: []
  parentID:
    type: string
    description: Parent ID
    verbs:
      - GET
      - PUT
      - POST
  parentLag:
    type: number
    description: Parent Lag
    verbs: []
  parentLagType:
    type: string
    description: Parent Lag Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - d
        description: Days
      - type: string
        enum:
          - c
        description: Calendar Days
      - type: string
        enum:
          - p
        description: Percent
      - type: string
        enum:
          - w
        description: Day of Week
      - type: string
        enum:
          - k
        description: Day of Week (Non-Zero)
  pendingCalculation:
    type: string
    description: pendingCalculation
    verbs: []
  pendingPredecessors:
    type: boolean
    description: Pending Predecessors
    verbs: []
  pendingUpdateMethods:
    type: array
    description: pendingUpdateMethods
    verbs: []
    items:
      type: string
    uniqueItems: true
  percentComplete:
    type: number
    description: Percent Complete
    verbs:
      - GET
      - PUT
      - POST
  personal:
    type: boolean
    description: Personal
    verbs:
      - GET
      - POST
  plannedBillableExpenseCost:
    type: number
    description: plannedBillableExpenseCost
    verbs: []
  plannedCompletionDate:
    type: string
    description: The date in which this task is planned or estimated to be completed
    verbs:
      - GET
  plannedCompletionDateExpression:
    type: string
    description: plannedCompletionDateExpression
    verbs: []
  plannedCost:
    type: number
    description: Planned Cost
    verbs: []
  plannedDateAlignment:
    type: integer
    description: Planned Date Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Will be done after the planned completion date
      - type: integer
        enum:
          - 1
        description: Will be done on the planned completion date
      - type: integer
        enum:
          - 2
        description: Will be done before the planned completion date
      - type: integer
        enum:
          - 3
        description: No set planned date
  plannedDuration:
    type: number
    description: Planned Duration
    verbs:
      - GET
      - PUT
      - POST
  plannedDurationMinutes:
    type: integer
    description: Planned Duration Minutes
    verbs: []
  plannedExpenseCost:
    type: number
    description: Planned Expense Cost
    verbs: []
  plannedHoursAlignment:
    type: integer
    description: Planned Hours Alignment
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: More hours than planned
      - type: integer
        enum:
          - 1
        description: Same hours as planned
      - type: integer
        enum:
          - 2
        description: Less hours than planned
      - type: integer
        enum:
          - 3
        description: No planned hours set
  plannedLaborCost:
    type: number
    description: Planned Labor Cost
    verbs: []
  plannedLaborRevenue:
    type: number
    description: Planned Labor Revenue
    verbs: []
  plannedNonBillableExpenseCost:
    type: number
    description: plannedNonBillableExpenseCost
    verbs: []
  plannedNonLaborCost:
    type: number
    description: Planned Non-Labor Cost
    verbs: []
  plannedNonLaborRevenue:
    type: number
    description: Planned Non-Labor Revenue
    verbs: []
  plannedRevenue:
    type: number
    description: Planned Revenue
    verbs: []
  plannedStartDate:
    type: string
    description: Planned Start Date
    verbs: []
  plannedStartDateExpression:
    type: string
    description: plannedStartDateExpression
    verbs: []
  predecessorExpression:
    type: string
    description: predecessorExpression
    verbs: []
  previousStatus:
    type: string
    description: Previous status
    verbs: []
    oneOf:
      - type: string
        enum:
          - NEW
        description: New
      - type: string
        enum:
          - INP
        description: In Progress
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - CPA
        description: Complete - Pending Approval
      - type: string
        enum:
          - CPI
        description: Complete - Pending Issues
  previousStatusLabel:
    type: string
    description: previousStatusLabel
    verbs: []
  priority:
    type: integer
    description: "Priority.  Use the enum value, not the description.  The only accepted values are: 0, 1, 2, 3, 4."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: integer
        enum:
          - 0
        description: None
      - type: integer
        enum:
          - 1
        description: Low
      - type: integer
        enum:
          - 2
        description: Normal
      - type: integer
        enum:
          - 3
        description: High
      - type: integer
        enum:
          - 4
        description: Urgent
  priorityColor:
    type: string
    description: priorityColor
    verbs: []
  progressStatus:
    type: string
    description: Progress Status
    verbs: []
    oneOf:
      - type: string
        enum:
          - LT
        description: Late
      - type: string
        enum:
          - "ON"
        description: On Time
      - type: string
        enum:
          - BH
        description: Behind
      - type: string
        enum:
          - RK
        description: At Risk
  projectedCompletionDate:
    type: string
    description: Projected Completion Date
    verbs: []
  projectedDurationMinutes:
    type: integer
    description: Projected Duration Minutes
    verbs: []
  projectedStartDate:
    type: string
    description: Projected Start Date
    verbs:
      - GET
      - PUT
      - POST
  projectedWorkPerDayList:
    type: array
    description: projectedWorkPerDayList
    verbs: []
    items:
      type: string
    uniqueItems: true
  projectID:
    type: string
    description: Project ID. Required when creating or updating a task.
    verbs:
      - GET
      - PUT
      - POST
  recurrenceNumber:
    type: integer
    description: Recurrence Number
    verbs: []
  recurrenceRuleID:
    type: string
    description: Recurrence Rule ID
    verbs: []
  referenceNumber:
    type: integer
    description: Reference Number
    verbs:
      - GET
  rejectionIssueID:
    type: string
    description: Rejection Issue ID
    verbs: []
  remainingDurationMinutes:
    type: integer
    description: Remaining Duration Minutes
    verbs: []
  resourceScope:
    type: string
    description: Resource Scope
    verbs: []
    oneOf:
      - type: string
        enum:
          - NONE
        description: None
      - type: string
        enum:
          - GLOB
        description: Global
  revenueType:
    type: string
    description: Revenue Type
    verbs: []
    oneOf:
      - type: string
        enum:
          - NOB
        description: Not Billable
      - type: string
        enum:
          - UHR
        description: User Hourly
      - type: string
        enum:
          - RHR
        description: Role Hourly
      - type: string
        enum:
          - FHR
        description: Fixed Hourly
      - type: string
        enum:
          - UHC
        description: User Hourly w/Cap
      - type: string
        enum:
          - RHC
        description: Role Hourly w/Cap
      - type: string
        enum:
          - UHF
        description: User Hourly Plus Fixed
      - type: string
        enum:
          - RHF
        description: Role Hourly Plus Fixed
      - type: string
        enum:
          - FRV
        description: Fixed Revenue
  roleID:
    type: string
    description: Role ID
    verbs: []
  rootGroupID:
    type: string
    description: rootGroupID
    verbs: []
  securityAncestorsDisabled:
    type: boolean
    description: securityAncestorsDisabled
    verbs: []
  securityRootID:
    type: string
    description: securityRootID
    verbs: []
  securityRootObjCode:
    type: string
    description: securityRootObjCode
    verbs: []
  showCommitDate:
    type: boolean
    description: showCommitDate
    verbs: []
  showCondition:
    type: boolean
    description: showCondition
    verbs: []
  showStatus:
    type: boolean
    description: showStatus
    verbs: []
  slackDate:
    type: string
    description: Slack Date
    verbs: []
  spi:
    type: number
    description: Schedule Performance Index
    verbs: []
  status:
    type: string
    description: "Status.  Use the enum value, not the description.  The only accepted values are: NEW, INP, CPL, CPA, CPI."
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - NEW
        description: New
      - type: string
        enum:
          - INP
        description: In Progress
      - type: string
        enum:
          - CPL
        description: Complete
      - type: string
        enum:
          - CPA
        description: Complete - Pending Approval
      - type: string
        enum:
          - CPI
        description: Complete - Pending Issues
  statusEquatesWith:
    type: string
    description: Status Equates With
    verbs: []
  statusLabel:
    type: string
    description: Status Label
    verbs: []
  statusUpdate:
    type: string
    description: Status Update
    verbs: []
  storyBoardOrder:
    type: integer
    description: storyBoardOrder
    verbs: []
  storyPoints:
    type: number
    description: Story Points
    verbs:
      - GET
      - PUT
      - POST
  submittedByID:
    type: string
    description: Submitted By ID
    verbs: []
  subscribersListString:
    type: string
    description: subscribersListString
    verbs: []
  taskConstraint:
    type: string
    description: Task Constraint
    verbs:
      - GET
      - PUT
      - POST
    oneOf:
      - type: string
        enum:
          - FIXT
        description: Fixed Dates
      - type: string
        enum:
          - MSO
        description: Must Start On
      - type: string
        enum:
          - MFO
        description: Must Finish On
      - type: string
        enum:
          - ASAP
        description: As Soon As Possible
      - type: string
        enum:
          - ALAP
        description: As Late As Possible
      - type: string
        enum:
          - EAT
        description: Earliest Available Time
      - type: string
        enum:
          - LAT
        description: Latest Available Time
      - type: string
        enum:
          - SNLT
        description: Start No Later Than
      - type: string
        enum:
          - SNET
        description: Start No Earlier Than
      - type: string
        enum:
          - FNLT
        description: Finish No Later Than
      - type: string
        enum:
          - FNET
        description: Finish No Earlier Than
  taskNumber:
    type: integer
    description: Task Number
    verbs:
      - GET
      - PUT
      - POST
  taskNumberPredecessorString:
    type: string
    description: taskNumberPredecessorString
    verbs: []
  teamID:
    type: string
    description: The ID of the team assigned to this Task
    verbs:
      - GET
      - PUT
      - POST
  templateTaskID:
    type: string
    description: Template Task ID
    verbs: []
  trackingMode:
    type: string
    description: Tracking Mode
    verbs: []
    oneOf:
      - type: string
        enum:
          - USER
        description: User Must Update
      - type: string
        enum:
          - ONTM
        description: Assume On Time
      - type: string
        enum:
          - IGNR
        description: Ignore Late Warnings
      - type: string
        enum:
          - AUTO
        description: Auto Complete
      - type: string
        enum:
          - PRED
        description: Predecessor
  URL:
    type: string
    description: URL
    verbs: []
  version:
    type: integer
    description: Version
    verbs: []
  wbs:
    type: string
    description: Work Breakdown Structure
    verbs: []
  work:
    type: number
    description: Planned Hours
    verbs: []
  workEffort:
    type: integer
    description: Work Effort
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 1
        description: Small
      - type: integer
        enum:
          - 2
        description: Medium
      - type: integer
        enum:
          - 3
        description: Large
  workPerDate:
    type: string
    description: workPerDate
    verbs: []
  workPerDayList:
    type: array
    description: workPerDayList
    verbs: []
    items:
      type: string
    uniqueItems: true
  workRequired:
    type: integer
    description: Planned Hours
    verbs: []
  workRequiredExpression:
    type: string
    description: workRequiredExpression
    verbs: []
  workUnit:
    type: string
    description: workUnit
    verbs: []
    oneOf:
      - type: string
        enum:
          - M
        description: Minutes
      - type: string
        enum:
          - H
        description: Hours
      - type: string
        enum:
          - D
        description: Days
      - type: string
        enum:
          - W
        description: Weeks
      - type: string
        enum:
          - T
        description: Months
      - type: string
        enum:
          - EM
        description: Elapsed Minutes
      - type: string
        enum:
          - EH
        description: Elapsed Hours
      - type: string
        enum:
          - ED
        description: Elapsed Days
      - type: string
        enum:
          - EW
        description: Elapsed Weeks
      - type: string
        enum:
          - ET
        description: Elapsed Months
required:
  - name
  - projectID

--- USER METADATA ---

domain_knowledge: |
  When searching for a user by name, be sure to check firstName, lastName, and name fields.  
  Use the OR operator because the name may not appear in all three fields.  
  Searching by email or username may also be used.
fields:
  accessLevelID:
    type: string
    description: The ID of the Access Level associated with this user
    verbs:
      - GET
  accessLevelIDBackup:
    type: string
    description: accessLevelIDBackup
    verbs: []
  actualDeactivationDate:
    type: string
    description: actualDeactivationDate
    verbs: []
  address:
    type: string
    description: Address
    verbs: []
  address2:
    type: string
    description: Address2
    verbs: []
  alignAccessType:
    type: string
    description: alignAccessType
    verbs: []
  apiKeyCreateDate:
    type: string
    description: apiKeyCreateDate
    verbs: []
  avatarDate:
    type: string
    description: Profile Photo Date
    verbs: []
  avatarDownloadURL:
    type: string
    description: avatarDownloadURL
    verbs: []
  avatarSize:
    type: integer
    description: Profile Photo Size
    verbs: []
  avatarX:
    type: integer
    description: Profile Photo X
    verbs: []
  avatarY:
    type: integer
    description: Profile Photo Y
    verbs: []
  billingPerHour:
    type: number
    description: Billing Per Hour
    verbs: []
  categoryID:
    type: string
    description: Category ID
    verbs: []
  city:
    type: string
    description: City
    verbs: []
  companyID:
    type: string
    description: Company ID
    verbs: []
  costPerHour:
    type: number
    description: Cost Per Hour
    verbs: []
  country:
    type: string
    description: Country
    verbs: []
  customerID:
    type: string
    description: Customer ID
    verbs: []
  customFormsDescriptions:
    type: string
    description: customFormsDescriptions
    verbs: []
  customFormsGroupIDs:
    type: string
    description: customFormsGroupIDs
    verbs: []
  customFormsIDs:
    type: string
    description: customFormsIDs
    verbs:
      - GET
  customFormsNames:
    type: string
    description: customFormsNames
    verbs:
      - GET
  defaultHourTypeID:
    type: string
    description: Default Hour Type ID
    verbs: []
  defaultInterface:
    type: integer
    description: Default Interface
    verbs: []
    oneOf:
      - type: integer
        enum:
          - 0
        description: Default
      - type: integer
        enum:
          - 1
        description: Project Management Home
      - type: integer
        enum:
          - 2
        description: TeamHome
      - type: integer
        enum:
          - 3
        description: Early Access
  delegationToID:
    type: string
    description: delegationToID
    verbs: []
  eauthUserID:
    type: string
    description: eauthUserID
    verbs: []
  emailAddr:
    type: string
    description: Email Address
    verbs:
      - GET
  enteredByID:
    type: string
    description: Entered By ID
    verbs:
      - GET
  entryDate:
    type: string
    description: Entry Date
    verbs:
      - GET
  extRefID:
    type: string
    description: External Reference ID
    verbs: []
  firstName:
    type: string
    description: First Name
    verbs:
      - GET
  fte:
    type: number
    description: FTE
    verbs: []
  hasAPIAccess:
    type: boolean
    description: Has API Access
    verbs: []
  hasDocuments:
    type: boolean
    description: Has Documents
    verbs: []
  hasNotes:
    type: boolean
    description: Has Notes
    verbs: []
  hasPassword:
    type: boolean
    description: hasPassword
    verbs: []
  hasProofLicense:
    type: boolean
    description: Has Proof License
    verbs: []
  hasQuicksilver:
    type: boolean
    description: hasQuicksilver
    verbs: []
  hasReservedTimes:
    type: boolean
    description: Has Time Off
    verbs: []
  homeGroupID:
    type: string
    description: The home group ID for this user
    verbs:
      - GET
  homeTeamID:
    type: string
    description: The home team ID for this user
    verbs:
      - GET
  ID:
    type: string
    description: User ID
    verbs:
      - GET
  isActive:
    type: boolean
    description: Is Active
    verbs: []
  isAdmin:
    type: boolean
    description: Is Admin
    verbs: []
  isBoxAuthenticated:
    type: boolean
    description: isBoxAuthenticated
    verbs: []
  isDropBoxAuthenticated:
    type: boolean
    description: isDropBoxAuthenticated
    verbs: []
  isGoogleAuthenticated:
    type: boolean
    description: isGoogleAuthenticated
    verbs: []
  isLimitedOrFullAdmin:
    type: boolean
    description: isLimitedOrFullAdmin
    verbs: []
  isSharePointAuthenticated:
    type: boolean
    description: isSharePointAuthenticated
    verbs: []
  isWebDAMAuthenticated:
    type: boolean
    description: isWebDAMAuthenticated
    verbs: []
  isWorkfrontDAMAuthenticated:
    type: boolean
    description: isWorkfrontDAMAuthenticated
    verbs: []
  lastEnteredNoteID:
    type: string
    description: Last Entered Note ID
    verbs: []
  lastLoginDate:
    type: string
    description: Last Login Date
    verbs:
      - GET
  lastName:
    type: string
    description: Last Name
    verbs:
      - GET
  lastNoteID:
    type: string
    description: Last Note ID
    verbs: []
  lastProofApprovalSync:
    type: string
    description: lastProofApprovalSync
    verbs: []
  lastUpdateDate:
    type: string
    description: Last Update Date
    verbs:
      - GET
  lastUpdatedByID:
    type: string
    description: Last Updated By ID
    verbs:
      - GET
  latestUpdateNoteID:
    type: string
    description: Latest Update Note ID
    verbs: []
  layoutTemplateID:
    type: string
    description: Layout Template ID
    verbs:
      - GET
  licenseType:
    type: string
    description: License type associated with this user
    verbs:
      - GET
    oneOf:
      - type: string
        enum:
          - F
        description: Standard
      - type: string
        enum:
          - C
        description: Light
      - type: string
        enum:
          - R
        description: Contributor
      - type: string
        enum:
          - X
        description: External
  licenseUpgradedDate:
    type: string
    description: Date Issued
    verbs: []
  locale:
    type: string
    description: Locale
    verbs: []
  loginBlocked:
    type: boolean
    description: Is Login Blocked
    verbs: []
  loginCount:
    type: integer
    description: Login Count
    verbs: []
  logTimeInDays:
    type: boolean
    description: logTimeInDays
    verbs: []
  managerID:
    type: string
    description: Manager ID
    verbs:
      - GET
  mobilePhoneNumber:
    type: string
    description: Mobile Phone
    verbs: []
  myInfo:
    type: string
    description: Talk to Me About
    verbs: []
  name:
    type: string
    description: Name
    verbs:
      - GET
  parentGroupIDs:
    type: array
    description: parentGroupIDs
    verbs: []
    items:
      type: string
    uniqueItems: true
  password:
    type: string
    description: Password
    verbs: []
  passwordDate:
    type: string
    description: Password Date
    verbs: []
  persona:
    type: string
    description: Persona
    verbs: []
  phoneExtension:
    type: string
    description: Extension
    verbs: []
  phoneNumber:
    type: string
    description: Phone number
    verbs: []
  portalProfileID:
    type: string
    description: Portal Profile ID
    verbs: []
  postalCode:
    type: string
    description: Postal Code
    verbs: []
  proofAccountRole:
    type: integer
    description: proofAccountRole
    verbs: []
  proofHQUsername:
    type: string
    description: proofHQUsername
    verbs: []
  proofsToApprove:
    type: array
    description: proofsToApprove
    verbs: []
    items:
      type: string
    uniqueItems: true
  receiveEmails:
    type: boolean
    description: receiveEmails
    verbs: []
  registrationExpireDate:
    type: string
    description: Registration Expiration Date
    verbs: []
  resetPasswordExpireDate:
    type: string
    description: Reset Password Expire Date
    verbs: []
  resourcePoolID:
    type: string
    description: Legacy Resource Pool ID
    verbs: []
  roleID:
    type: string
    description: Role ID
    verbs:
      - GET
  scheduledDeactivationDate:
    type: string
    description: Scheduled Deactivation Date
    verbs: []
  scheduleID:
    type: string
    description: Schedule ID
    verbs: []
  ssoAccessOnly:
    type: boolean
    description: SSO Access Only
    verbs: []
  ssoUsername:
    type: string
    description: SSO Username
    verbs: []
  state:
    type: string
    description: State
    verbs: []
  statusUpdate:
    type: string
    description: Status Update
    verbs: []
  teamsListString:
    type: string
    description: teamsListString
    verbs: []
  timesheetProfileID:
    type: string
    description: Timesheet Profile ID
    verbs: []
  timeZone:
    type: string
    description: Time Zone
    verbs: []
  title:
    type: string
    description: Title
    verbs:
      - GET
  uiTemplateID:
    type: string
    description: Layout Template ID
    verbs: []
  username:
    type: string
    description: Username
    verbs:
      - GET
  uumUserID:
    type: string
    description: uumUserID
    verbs: []
  webDAVProfile:
    type: array
    description: Web DAV Profile
    verbs: []
    items:
      oneOf:
        - type: string
          enum:
            - ALL
          description: All Projects
        - type: string
          enum:
            - CUR
          description: Current Projects
        - type: string
          enum:
            - FAV
          description: Favorite Projects
    uniqueItems: true
  workHoursPerDay:
    type: number
    description: workHoursPerDay
    verbs: []
  workTime:
    type: number
    description: Work Time
    verbs: []
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
I'll help you with defining the correct call with the correct ObjCode, Fields, and Filters.

<thinking>
I need to understand the user's request and determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy based on the metadata provided).
</thinking>

<answer>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/workfront_api_tasks')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file with Workfront data')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'workfront_api_tasks'

    # Load the Workfront dataset
    print("Loading Workfront API dataset...")
    raw_samples = load_workfront_data(args.json_file)
    
    # Split into train and test
    total_samples = len(raw_samples)
    train_size = int(total_samples * args.train_split)
    
    train_samples = raw_samples[:train_size]
    test_samples = raw_samples[train_size:]

    def process_samples(samples, split):
        processed_data = []
        for idx, sample in enumerate(samples):
            question = make_prefix(sample['question'], template_type=args.template_type)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "api_planning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "correct_answer": sample['answer'],
                        "question": sample['question'],
                        "template_type": sample['template_type'],
                        "original_response": sample.get('original_response', {})
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            processed_data.append(data)
        return processed_data
    
    train_dataset = Dataset.from_list(process_samples(train_samples, 'train'))
    test_dataset = Dataset.from_list(process_samples(test_samples, 'test'))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving datasets to {local_dir}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print("Sample data:")
    print("Question:", train_samples[0]['question'])
    print("Answer:", train_samples[0]['answer'])