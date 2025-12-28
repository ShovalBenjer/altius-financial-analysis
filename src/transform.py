import polars as pl, pandas as pd, duckdb, os, sys, json, time, plotly.express as px
from loguru import logger

def main():
    logger.configure(handlers=[{"sink": sys.stderr, "format": "<green>{time}</green> <level>{message}</level>"}])
    CONF = json.load(open("config.json")) if os.path.exists("config.json") else {"fx_rates":{"USD":1.0},"data_path":"data.xlsx"}
    RAW_FILE, FX_RATES = CONF.get("data_path", "Exercise Data Analyst Junior.xlsx"), CONF.get("fx_rates", {"USD":1.0})
    if not os.path.exists(RAW_FILE): logger.error(f"Source file not found: {RAW_FILE}"); sys.exit(1)
    
    logger.info("Loading and cleaning Excel sheets...")
    xls, dfs = pd.read_excel(RAW_FILE, sheet_name=None, dtype=str), {}
    for k in ['Assets Data', 'Files Data', 'NAV Data']:
        d = xls.get(k, pd.DataFrame())
        d.columns = [c.strip().lower().replace(' ', '_') for c in d.columns]
        d.rename(columns={'deal_nam':'deal_name','managem':'management_fees','asset_clas':'asset_class','type':'measure','value':'amount','venture_d_commitment':'commitment'}, inplace=True)
        if 'deal_name' in d.columns: d['deal_name'] = d['deal_name'].replace(['nan','None',''], None).ffill()
        dfs[k] = pl.from_pandas(d)
        
    logger.info("Transforming data...")
    assets = dfs['Assets Data'].with_columns([pl.col("commitment").str.replace_all(r"[$,]", "").str.strip_chars().replace("nan","0").cast(pl.Float64, strict=False).fill_null(0), pl.col("deal_name").str.strip_chars().str.to_uppercase(), pl.col("currency").str.strip_chars().str.to_uppercase().fill_null("USD")]).filter(pl.col("deal_name").is_not_null())
    clean_cols = [pl.col("amount").str.replace_all(r"[$,]", "").str.strip_chars().replace("nan","0").cast(pl.Float64, strict=False).fill_null(0), pl.col("date").str.slice(0,10).str.to_date(strict=False), pl.col("deal_name").str.strip_chars().str.to_uppercase()]
    files = dfs['Files Data'].with_columns(clean_cols).filter(pl.col("date").is_not_null()).with_columns(pl.col("measure").replace({"Provided NAV":"Estimated NAV","Chargeback CC":"Chargeback Capital Call"}))
    nav = dfs['NAV Data'].with_columns(clean_cols).filter(pl.col("date").is_not_null()).with_columns([pl.col("measure").replace({"Provided NAV":"Estimated NAV"}), pl.lit("Fund Level").alias("investor") if "investor" not in dfs['NAV Data'].columns else pl.col("investor")])
    curr_map = assets.select(["deal_name","currency"]).unique()
    measures = pl.concat([files.select(["deal_name","date","investor","measure","amount"]), nav.select(["deal_name","date","investor","measure","amount"])]).unique().join(curr_map, on="deal_name", how="left").with_columns(pl.when(pl.col("measure").str.to_lowercase().str.starts_with("commit")).then(pl.lit("Commitment")).otherwise(pl.col("measure")).alias("measure"))
    measures = measures.with_columns(pl.col("currency").replace(FX_RATES).fill_null(1.0).cast(pl.Float64).alias("fx_rate")).with_columns((pl.col("amount")*pl.col("fx_rate")).alias("amount_usd")).sort(["deal_name","date"])
    agg = measures.group_by("deal_name").agg([pl.col("amount_usd").filter(pl.col("measure")=="Capital Call").sum().alias("Capital Calls"), pl.col("amount_usd").filter(pl.col("measure")=="Commitment").sum().alias("calc_commitment"), pl.col("date").filter(pl.col("measure")=="Capital Call").min().dt.year().alias("Vintage"), pl.col("date").filter(pl.col("measure")=="Capital Call").min().dt.strftime("%m/%d/%Y").alias("Commitment Date")])
    assets_final = assets.with_columns(pl.col("currency").replace(FX_RATES).fill_null(1.0).cast(pl.Float64).alias("fx_rate")).with_columns((pl.col("commitment")*pl.col("fx_rate")).alias("reported_commitment_usd"))
    metadata = assets_final.join(agg, on="deal_name", how="left").select([pl.col("deal_name").alias("Deal Name"), pl.col("management_fees").alias("Management Fees"), "Commitment Date", "Vintage", "currency", pl.col("geography").alias("Geography"), pl.col("asset_class").alias("Asset Class"), pl.col("underlying").alias("Underlying"), pl.col("pe_tags").alias("PE Tags"), pl.col("calc_commitment").alias("Commitment"), "Capital Calls"])
    measures_out = measures.select([pl.col("deal_name").alias("Deal Name"), pl.col("date").dt.strftime("%Y-%m-%d").alias("Date"), pl.col("investor").alias("Investor"), pl.col("measure").alias("Measure"), pl.col("amount_usd").alias("Amount")])
    
    logger.info("Auditing...")
    duckdb.register("meta", assets_final); duckdb.register("aggs", agg)
    audit_df = duckdb.sql("SELECT m.deal_name, m.reported_commitment_usd AS Rep, a.calc_commitment AS Calc, ABS(m.reported_commitment_usd - a.calc_commitment) AS Delta FROM meta m LEFT JOIN aggs a ON m.deal_name=a.deal_name ORDER BY Delta DESC").pl()
    if len(audit_df.filter(pl.col("Delta") > 10.0)) > 0: logger.warning(f"Mismatch found:\n{audit_df}")
    else: logger.success("Audit OK.")
    
    try:
        sb_data = metadata.with_columns(pl.col("Vintage").fill_null(0).cast(pl.Int64).cast(pl.Utf8).replace("0","Unknown")).to_pandas()
        px.sunburst(sb_data, path=["Asset Class", "Geography", "Vintage"], values="Commitment", title="Portfolio Concentration Risk (by Commitment)", width=800, height=800).write_html("portfolio_sunburst.html")
        logger.success("Saved sunburst to portfolio_sunburst.html")
    except Exception as e: logger.error(f"Plot error: {e}")
    
    metadata.write_csv("metadata.csv"); measures_out.write_csv("measures.csv")
    logger.success(f"Done. Wrote metadata.csv ({len(metadata)} deals) and measures.csv ({len(measures_out)} records).")

if __name__=="__main__":
    main()

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def getLLMDescription(row, market_ctx, client, model_id):
    """Generate a concise, context-aware financial assessment for a deal using an LLM."""
    age=2025-int(row.get('Vintage', 2020) or 2020)
    stage="Harvesting" if age>7 else "Investing" if age<3 else "Maturing"
    msgs=[{"role":"system","content":"You are a Senior Risk Analyst. Write a 1-sentence dense financial assessment."},
          {"role":"user","content":f"Context:{market_ctx}\nDeal:{row['Deal Name']} ({row['Asset Class']} - {row['Underlying']})\nProfile:Vintage {row['Vintage']} ({stage} Phase)|Geo:{row['Geography']}\nPerformance:{row.get('NAV_Growth','N/A')} NAV Growth\nInstruction:Assess performance vs market context. Be brief."}]
    return client.chat_completion(model=model_id,messages=msgs,max_tokens=100,temperature=0.3).choices[0].message.content.strip()
