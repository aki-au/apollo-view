import sqlite3
import json
import os
from pathlib import Path

# CONFIG
DATA_DIR = Path("./structured-data")
DB_PATH = Path("./calpers.db")


def load_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_tables(conn):
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS plan_financials (
            plan_name TEXT PRIMARY KEY,
            total_aum_billions REAL,
            funded_status_pct REAL,
            fy2025_return_pct REAL,
            discount_rate_pct REAL,
            return_1yr_pct REAL,
            return_3yr_pct REAL,
            return_5yr_pct REAL,
            return_10yr_pct REAL,
            return_20yr_pct REAL,
            total_members INTEGER,
            annual_benefit_payments_billions REAL,
            employer_contribution_rate_pct REAL,
            last_actuarial_review_date TEXT,
            next_actuarial_review_date TEXT,
            investment_consultants TEXT,
            fiscal_year_end TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS manager_roster (
            manager_id TEXT PRIMARY KEY,
            manager_name TEXT,
            strategy_type TEXT,
            aum_managed_billions REAL,
            mandate_start_date TEXT,
            contract_renewal_date TEXT,
            benchmark TEXT,
            fee_bps REAL,
            fee_dollars_millions REAL,
            mandate_status TEXT,
            internal_or_external TEXT,
            primary_contact TEXT,
            last_review_date TEXT,
            notes TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            manager_id TEXT PRIMARY KEY,
            manager_name TEXT,
            strategy_type TEXT,
            return_1yr_pct REAL,
            return_3yr_pct REAL,
            return_5yr_pct REAL,
            benchmark_return_1yr_pct REAL,
            benchmark_return_3yr_pct REAL,
            benchmark_return_5yr_pct REAL,
            alpha_1yr_bps REAL,
            alpha_3yr_bps REAL,
            alpha_5yr_bps REAL,
            sharpe_ratio REAL,
            tracking_error_pct REAL,
            information_ratio REAL,
            performance_vs_peers TEXT,
            last_performance_review_date TEXT,
            FOREIGN KEY (manager_id) REFERENCES manager_roster(manager_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS board_members (
            member_id TEXT PRIMARY KEY,
            name TEXT,
            role TEXT,
            appointment_type TEXT,
            professional_background TEXT,
            tenure_years INTEGER,
            term_expiry_date TEXT,
            committee_memberships TEXT,
            known_priorities TEXT,
            finance_expertise_level TEXT,
            typical_question_focus TEXT,
            notes TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS key_dates (
            date_id TEXT PRIMARY KEY,
            event_type TEXT,
            event_name TEXT,
            date TEXT,
            related_manager_id TEXT,
            priority TEXT,
            description TEXT,
            action_required INTEGER,
            action_description TEXT,
            FOREIGN KEY (related_manager_id) REFERENCES manager_roster(manager_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS past_meeting_notes (
            note_id TEXT PRIMARY KEY,
            meeting_date TEXT,
            meeting_type TEXT,
            calpers_attendees TEXT,
            apex_attendees TEXT,
            location TEXT,
            strategy_discussed TEXT,
            key_discussion_points TEXT,
            board_concerns_raised TEXT,
            action_items TEXT,
            outcome TEXT,
            follow_up_required INTEGER,
            internal_notes TEXT,
            sentiment_score INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS competitive_intelligence (
            competitor_id TEXT PRIMARY KEY,
            firm_name TEXT,
            strategy_focus TEXT,
            estimated_calpers_aum_billions REAL,
            is_current_calpers_manager INTEGER,
            known_fee_range_bps TEXT,
            esg_rating TEXT,
            recent_mandate_wins TEXT,
            recent_mandate_losses TEXT,
            known_strengths TEXT,
            known_weaknesses TEXT,
            threat_level TEXT,
            notes TEXT
        )
    """)

    conn.commit()
    print("Tables created")


def insert_financials(conn, data):
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO plan_financials VALUES (
            :plan_name, :total_aum_billions, :funded_status_pct,
            :fy2025_return_pct, :discount_rate_pct,
            :return_1yr_pct, :return_3yr_pct, :return_5yr_pct,
            :return_10yr_pct, :return_20yr_pct,
            :total_members, :annual_benefit_payments_billions,
            :employer_contribution_rate_pct,
            :last_actuarial_review_date, :next_actuarial_review_date,
            :investment_consultants, :fiscal_year_end
        )
    """, {
        **data,
        "investment_consultants": json.dumps(data["investment_consultants"])
    })
    conn.commit()
    print("Loaded plan_financials")


def insert_managers(conn, data):
    c = conn.cursor()
    for row in data:
        c.execute("""
            INSERT OR REPLACE INTO manager_roster VALUES (
                :manager_id, :manager_name, :strategy_type,
                :aum_managed_billions, :mandate_start_date,
                :contract_renewal_date, :benchmark, :fee_bps,
                :fee_dollars_millions, :mandate_status,
                :internal_or_external, :primary_contact,
                :last_review_date, :notes
            )
        """, row)
    conn.commit()
    print(f"Loaded manager_roster ({len(data)})")


def insert_performance(conn, data):
    c = conn.cursor()
    for row in data:
        c.execute("""
            INSERT OR REPLACE INTO performance VALUES (
                :manager_id, :manager_name, :strategy_type,
                :return_1yr_pct, :return_3yr_pct, :return_5yr_pct,
                :benchmark_return_1yr_pct, :benchmark_return_3yr_pct,
                :benchmark_return_5yr_pct,
                :alpha_1yr_bps, :alpha_3yr_bps, :alpha_5yr_bps,
                :sharpe_ratio, :tracking_error_pct, :information_ratio,
                :performance_vs_peers, :last_performance_review_date
            )
        """, row)
    conn.commit()
    print(f"Loaded performance ({len(data)})")


def insert_board_members(conn, data):
    c = conn.cursor()
    for row in data:
        c.execute("""
            INSERT OR REPLACE INTO board_members VALUES (
                :member_id, :name, :role, :appointment_type,
                :professional_background, :tenure_years, :term_expiry_date,
                :committee_memberships, :known_priorities,
                :finance_expertise_level, :typical_question_focus, :notes
            )
        """, {
            **row,
            "committee_memberships": json.dumps(row["committee_memberships"]),
            "known_priorities": json.dumps(row["known_priorities"])
        })
    conn.commit()
    print(f"Loaded board_members ({len(data)})")


def insert_key_dates(conn, data):
    c = conn.cursor()
    for row in data:
        c.execute("""
            INSERT OR REPLACE INTO key_dates VALUES (
                :date_id, :event_type, :event_name, :date,
                :related_manager_id, :priority, :description,
                :action_required, :action_description
            )
        """, {
            **row,
            "action_required": int(bool(row["action_required"]))
        })
    conn.commit()
    print(f"Loaded key_dates ({len(data)})")


def insert_meeting_notes(conn, data):
    c = conn.cursor()
    for row in data:
        c.execute("""
            INSERT OR REPLACE INTO past_meeting_notes VALUES (
                :note_id, :meeting_date, :meeting_type,
                :calpers_attendees, :apex_attendees, :location,
                :strategy_discussed, :key_discussion_points,
                :board_concerns_raised, :action_items,
                :outcome, :follow_up_required, :internal_notes,
                :sentiment_score
            )
        """, {
            **row,
            "calpers_attendees": json.dumps(row["calpers_attendees"]),
            "apex_attendees": json.dumps(row["apex_attendees"]),
            "key_discussion_points": json.dumps(row["key_discussion_points"]),
            "board_concerns_raised": json.dumps(row["board_concerns_raised"]),
            "action_items": json.dumps(row["action_items"]),
            "follow_up_required": int(bool(row["follow_up_required"]))
        })
    conn.commit()
    print(f"Loaded past_meeting_notes ({len(data)})")


def insert_competitive_intelligence(conn, data):
    c = conn.cursor()
    for row in data:
        c.execute("""
            INSERT OR REPLACE INTO competitive_intelligence VALUES (
                :competitor_id, :firm_name, :strategy_focus,
                :estimated_calpers_aum_billions,
                :is_current_calpers_manager, :known_fee_range_bps,
                :esg_rating, :recent_mandate_wins, :recent_mandate_losses,
                :known_strengths, :known_weaknesses, :threat_level, :notes
            )
        """, {
            **row,
            "strategy_focus": json.dumps(row["strategy_focus"]),
            "recent_mandate_wins": json.dumps(row["recent_mandate_wins"]),
            "recent_mandate_losses": json.dumps(row["recent_mandate_losses"]),
            "known_strengths": json.dumps(row["known_strengths"]),
            "known_weaknesses": json.dumps(row["known_weaknesses"]),
            "is_current_calpers_manager": int(bool(row["is_current_calpers_manager"]))
        })
    conn.commit()
    print(f"Loaded competitive_intelligence ({len(data)})")


def verify(conn):
    c = conn.cursor()

    tables = [
        "plan_financials",
        "manager_roster",
        "performance",
        "board_members",
        "key_dates",
        "past_meeting_notes",
        "competitive_intelligence"
    ]

    print("\nTable counts:")
    for t in tables:
        c.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"{t}: {c.fetchone()[0]}")

    print("\nManagers not active:")
    c.execute("""
        SELECT manager_name, mandate_status
        FROM manager_roster
        WHERE mandate_status != 'Active'
    """)
    for row in c.fetchall():
        print(row)

    print("\nNegative alpha managers:")
    c.execute("""
        SELECT m.manager_name, p.alpha_1yr_bps
        FROM performance p
        JOIN manager_roster m ON p.manager_id = m.manager_id
        WHERE p.alpha_1yr_bps < 0
    """)
    for row in c.fetchall():
        print(row)

    print("\nRFP deadlines:")
    c.execute("""
        SELECT event_name, date
        FROM key_dates
        WHERE event_type = 'RFP Deadline'
    """)
    for row in c.fetchall():
        print(row)

    print("\nDatabase ready")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Existing database removed")

    conn = connect_db(DB_PATH)

    try:
        create_tables(conn)

        insert_financials(conn, load_json("calpERS_financials.json"))
        insert_managers(conn, load_json("calpERS_fixed_income_managers.json"))
        insert_performance(conn, load_json("calpERS_fixed_income_performance_history.json"))
        insert_board_members(conn, load_json("calpERS_board_members.json"))
        insert_key_dates(conn, load_json("calpERS_key_dates_calendar.json"))
        insert_meeting_notes(conn, load_json("calpERS_internal_meeting_notes.json"))
        insert_competitive_intelligence(conn, load_json("calpERS_competitive_intelligence_summary.json"))

        verify(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()