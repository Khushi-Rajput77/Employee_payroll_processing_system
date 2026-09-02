#!/usr/bin/env python3
"""
Employee Payroll Processing System
===================================

Reads employee records from a CSV file and computes gross salary, tax,
deductions, and net salary for each employee. Calculation rules differ
by employee type (full-time, manager, contractor, intern).

Input CSV columns (required):
    employee_id,name,department,basic_salary,bonus,tax

Optional column:
    employee_type   -> one of: full-time, manager, contractor, intern
                        (defaults to "full-time" if the column/value is missing)

The "tax" column is interpreted differently per employee type — see each
class's docstring below for exactly how it is used.

Usage:
    python payroll_system.py employees.csv -o payroll_output.csv
"""

import csv
import sys
import argparse
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import List, Dict
from abc import ABC, abstractmethod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def money(value) -> Decimal:
    """Convert any numeric-ish value to a Decimal rounded to 2 places."""
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        raise ValueError(f"'{value}' is not a valid number")


# --------------------------------------------------------------------------- #
# Employee hierarchy
# --------------------------------------------------------------------------- #

class Employee(ABC):
    """
    Abstract base class for every employee type.

    Subclasses must implement calculate_gross(), calculate_tax(), and
    calculate_deductions(). net salary is always:

        net_salary = gross_salary - tax - deductions
    """

    PROFESSIONAL_TAX = Decimal("200.00")   # flat statutory tax, full-timers & managers only
    PF_RATE = Decimal("0.12")              # employee Provident Fund contribution (12% of basic)

    def __init__(self, employee_id: str, name: str, department: str,
                 basic_salary, bonus, tax):
        if not employee_id or not name:
            raise ValueError("employee_id and name are required")
        self.employee_id = employee_id
        self.name = name
        self.department = department
        self.basic_salary = money(basic_salary)
        self.bonus = money(bonus)
        self.input_tax = money(tax)     # raw "tax" field from the CSV (meaning varies by type)

        if self.basic_salary < 0 or self.bonus < 0:
            raise ValueError("basic_salary and bonus cannot be negative")

    @property
    def employee_type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def calculate_gross(self) -> Decimal:
        """Basic + bonus + any type-specific allowances."""

    @abstractmethod
    def calculate_tax(self) -> Decimal:
        """Income tax owed, computed however this employee type is taxed."""

    @abstractmethod
    def calculate_deductions(self) -> Decimal:
        """Non-tax deductions (PF, professional tax, etc.), excluding income tax."""

    def calculate_net(self) -> Decimal:
        return money(self.calculate_gross() - self.calculate_tax() - self.calculate_deductions())

    def payslip_row(self) -> Dict[str, str]:
        gross = self.calculate_gross()
        tax = self.calculate_tax()
        deductions = self.calculate_deductions()
        net = money(gross - tax - deductions)
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "department": self.department,
            "employee_type": self.employee_type,
            "basic_salary": f"{self.basic_salary:.2f}",
            "bonus": f"{self.bonus:.2f}",
            "gross_salary": f"{gross:.2f}",
            "tax": f"{tax:.2f}",
            "deductions": f"{deductions:.2f}",
            "net_salary": f"{net:.2f}",
        }


class FullTimeEmployee(Employee):
    """
    Standard salaried staff.

    - gross  = basic_salary + bonus
    - tax    = "tax" column read as a PERCENTAGE rate, applied to gross
    - deductions = 12% Provident Fund (on basic) + flat professional tax
    """

    def calculate_gross(self) -> Decimal:
        return money(self.basic_salary + self.bonus)

    def calculate_tax(self) -> Decimal:
        rate = self.input_tax / Decimal("100")
        return money(self.calculate_gross() * rate)

    def calculate_deductions(self) -> Decimal:
        pf = self.basic_salary * self.PF_RATE
        return money(pf + self.PROFESSIONAL_TAX)


class Manager(FullTimeEmployee):
    """
    Full-time employee plus a management allowance.

    - gross = basic_salary + bonus + 15% management allowance (on basic)
    - tax / deductions: same rules as FullTimeEmployee, applied to the
      (higher) manager gross salary
    """

    MANAGEMENT_ALLOWANCE_RATE = Decimal("0.15")

    def calculate_gross(self) -> Decimal:
        allowance = self.basic_salary * self.MANAGEMENT_ALLOWANCE_RATE
        return money(self.basic_salary + self.bonus + allowance)


class Contractor(Employee):
    """
    Paid per invoiced amount; no bonus/PF/professional tax.

    - gross = basic_salary only (bonus is ignored — contractors are
      paid strictly on contracted amount)
    - tax   = "tax" column read as a TDS PERCENTAGE, applied to gross
    - deductions = 0 (no PF, no professional tax)
    """

    def calculate_gross(self) -> Decimal:
        return money(self.basic_salary)

    def calculate_tax(self) -> Decimal:
        rate = self.input_tax / Decimal("100")
        return money(self.calculate_gross() * rate)

    def calculate_deductions(self) -> Decimal:
        return Decimal("0.00")


class Intern(Employee):
    """
    Stipend-based; no PF/professional tax; tax-exempt below a threshold.

    - gross = basic_salary (stipend) + bonus
    - tax   = 0 if gross <= TAX_EXEMPT_THRESHOLD, otherwise "tax" column
              read as a percentage rate applied to gross
    - deductions = 0
    """

    TAX_EXEMPT_THRESHOLD = Decimal("15000.00")

    def calculate_gross(self) -> Decimal:
        return money(self.basic_salary + self.bonus)

    def calculate_tax(self) -> Decimal:
        gross = self.calculate_gross()
        if gross <= self.TAX_EXEMPT_THRESHOLD:
            return Decimal("0.00")
        rate = self.input_tax / Decimal("100")
        return money(gross * rate)

    def calculate_deductions(self) -> Decimal:
        return Decimal("0.00")


EMPLOYEE_TYPE_MAP = {
    "full-time": FullTimeEmployee,
    "fulltime": FullTimeEmployee,
    "full_time": FullTimeEmployee,
    "manager": Manager,
    "contractor": Contractor,
    "intern": Intern,
}


# ------------------------------------------------------------------------- #
# Processor
# ------------------------------------------------------------------------- #

class PayrollProcessor:
    """Loads employees from CSV, runs calculations, and writes/reports results."""

    REQUIRED_COLUMNS = {"employee_id", "name", "department", "basic_salary", "bonus", "tax"}

    def __init__(self):
        self.employees: List[Employee] = []
        self.errors: List[str] = []

    def load_csv(self, path: str, type_column: str = "employee_type") -> None:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = self.REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

            for line_no, row in enumerate(reader, start=2):  # header is line 1
                try:
                    emp_type_raw = (row.get(type_column) or "full-time").strip().lower()
                    emp_cls = EMPLOYEE_TYPE_MAP.get(emp_type_raw)
                    if emp_cls is None:
                        self.errors.append(
                            f"Row {line_no}: unknown employee_type '{emp_type_raw}', row skipped"
                        )
                        continue

                    employee = emp_cls(
                        employee_id=row["employee_id"].strip(),
                        name=row["name"].strip(),
                        department=row["department"].strip(),
                        basic_salary=row["basic_salary"],
                        bonus=row["bonus"],
                        tax=row["tax"],
                    )
                    self.employees.append(employee)
                except Exception as exc:
                    self.errors.append(f"Row {line_no}: {exc}, row skipped")

    def process(self) -> List[Dict[str, str]]:
        return [emp.payslip_row() for emp in self.employees]

    def write_csv(self, out_path: str) -> None:
        rows = self.process()
        fieldnames = [
            "employee_id", "name", "department", "employee_type",
            "basic_salary", "bonus", "gross_salary", "tax", "deductions", "net_salary",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def summary(self) -> Dict[str, Decimal]:
        totals = {"gross_salary": Decimal("0"), "tax": Decimal("0"),
                  "deductions": Decimal("0"), "net_salary": Decimal("0")}
        for row in self.process():
            for key in totals:
                totals[key] += Decimal(row[key])
        return totals

    def summary_by_type(self) -> Dict[str, Dict]:
        by_type: Dict[str, Dict] = {}
        for row in self.process():
            t = row["employee_type"]
            bucket = by_type.setdefault(t, {"count": 0, "names": [], "gross_salary": Decimal("0"),
                                             "tax": Decimal("0"), "deductions": Decimal("0"),
                                             "net_salary": Decimal("0")})
            bucket["count"] += 1
            bucket["names"].append(row["name"])
            for key in ("gross_salary", "tax", "deductions", "net_salary"):
                bucket[key] += Decimal(row[key])
        return by_type


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Employee Payroll Processing System")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", default="payroll_output.csv",
                         help="Path to write the computed payroll CSV (default: payroll_output.csv)")
    args = parser.parse_args()

    processor = PayrollProcessor()
    try:
        processor.load_csv(args.input_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    processor.write_csv(args.output)

    if processor.errors:
        print("Warnings:")
        for err in processor.errors:
            print(f"  - {err}")

    totals = processor.summary()
    print(f"\nProcessed {len(processor.employees)} employee(s).")
    print(f"  Total Gross Salary : {totals['gross_salary']:.2f}")
    print(f"  Total Tax          : {totals['tax']:.2f}")
    print(f"  Total Deductions   : {totals['deductions']:.2f}")
    print(f"  Total Net Salary   : {totals['net_salary']:.2f}")

    by_type = processor.summary_by_type()
    if by_type:
        print("\nBreakdown by employee type:")
        for emp_type, stats in sorted(by_type.items()):
            print(f"  {emp_type} ({stats['count']}):")
            print(f"      Names        : {', '.join(stats['names'])}")
            print(f"      Gross Salary : {stats['gross_salary']:.2f}")
            print(f"      Tax          : {stats['tax']:.2f}")
            print(f"      Deductions   : {stats['deductions']:.2f}")
            print(f"      Net Salary   : {stats['net_salary']:.2f}")
    print(f"\nDetailed results written to: {args.output}")


if __name__ == "__main__":
    main()