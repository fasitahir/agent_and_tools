# bookme_scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import time
import os

def scrape_all_bookme_data():
    driver = webdriver.Chrome()
    all_data = {}
    try:
        driver.get("https://bookme.pk")
        time.sleep(10)
        categories = driver.find_elements(By.CSS_SELECTOR, "a")
        category_links = {}
        for cat in categories:
            name = cat.text.strip()
            href = cat.get_attribute('href')
            if name and href and 'bookme.pk' in href and name.lower() not in category_links:
                if not any(x in name.lower() for x in ["login", "sign up", "signup", "sign in", "help", "contact", "about", "faq", "blog", "announcement", "terms", "privacy", "newsletter", "app", "company"]):
                    if not href.endswith("#") and not href.endswith("/"):
                        category_links[name] = href

        for name, link in category_links.items():
            driver.get(link)
            time.sleep(5)
            section_data = []

            if "bus" in name.lower() or "flight" in name.lower():
                try:
                    origin_options = set()
                    destination_options = set()
                    origin_selects = driver.find_elements(By.CSS_SELECTOR, 'select[id*="origin"], select[id*="from"]')
                    dest_selects = driver.find_elements(By.CSS_SELECTOR, 'select[id*="destination"], select[id*="to"]')
                    if origin_selects:
                        for opt in origin_selects[0].find_elements(By.TAG_NAME, 'option'):
                            val = opt.text.strip()
                            if val and val.lower() not in ["from", "origin"]:
                                origin_options.add(val)
                    if dest_selects:
                        for opt in dest_selects[0].find_elements(By.TAG_NAME, 'option'):
                            val = opt.text.strip()
                            if val and val.lower() not in ["to", "destination"]:
                                destination_options.add(val)
                    if not origin_options:
                        origin_options = {"Lahore", "Karachi", "Islamabad"}
                    if not destination_options:
                        destination_options = {"Lahore", "Karachi", "Islamabad"}
                    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

                    for origin in list(origin_options)[:3]:
                        for dest in list(destination_options)[:3]:
                            if origin == dest:
                                continue
                            try:
                                origin_input = driver.find_element(By.CSS_SELECTOR, '[id*="origin"], [id*="from"]')
                                destination_input = driver.find_element(By.CSS_SELECTOR, '[id*="destination"], [id*="to"]')
                                date_input = driver.find_element(By.CSS_SELECTOR, '[id*="date"], [id*="departure"]')
                                origin_input.clear()
                                origin_input.send_keys(origin)
                                destination_input.clear()
                                destination_input.send_keys(dest)
                                date_input.clear()
                                date_input.send_keys(tomorrow)
                                time.sleep(1)
                                search_btn = driver.find_element(By.CSS_SELECTOR, '[type="submit"], [id*="search"]')
                                search_btn.click()
                                if "bus" in name.lower():
                                    WebDriverWait(driver, 10).until(
                                        EC.presence_of_element_located((By.CLASS_NAME, "bus-item"))
                                    )
                                    items = driver.find_elements(By.CLASS_NAME, "bus-item")
                                    for item in items:
                                        try:
                                            time_ = item.find_element(By.CLASS_NAME, "time").text
                                            price = item.find_element(By.CLASS_NAME, "price").text
                                            company = item.find_element(By.CLASS_NAME, "company").text
                                            section_data.append(f"{origin} to {dest} | {company} | {time_} | {price}")
                                        except Exception:
                                            continue
                            except Exception as e:
                                section_data.append(f"Error scraping {origin} to {dest}: {e}")
                except Exception as e:
                    section_data.append(f"Error scraping routes: {e}")

            else:
                items = driver.find_elements(By.CSS_SELECTOR, '[class*="card"], [class*="item"], [class*="row"], [class*="col"]')
                for item in items:
                    try:
                        text = item.text.strip()
                        if text and text not in section_data:
                            section_data.append(text)
                    except Exception:
                        continue

            all_data[name] = section_data[:70]
    except Exception as e:
        print(f"Error scraping Bookme: {e}")
    finally:
        driver.quit()

    return all_data

def format_all_data(data):
    formatted = []
    for section, items in data.items():
        formatted.append(f"{section} (up to 10 items):\n" + "\n".join(items[:10]))
    return "\n\n".join(formatted)

def get_saved_context():
    if os.path.exists("bookme_scraped_data.txt"):
        with open("bookme_scraped_data.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    return None
