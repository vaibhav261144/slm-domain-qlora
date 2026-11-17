"""
Dataset Generator & Preprocessing Pipeline for Bilingual Domain QA (English + Hindi).
Generates curated domain examples, removes duplicates, validates schema,
and produces leak-free train/val/test splits grouped by source document.
"""

import json
from pathlib import Path
from src.data.preprocessor import deduplicate_dataset
from src.data.splitter import split_dataset, save_splits_to_disk
from src.data.dataset_loader import validate_chat_schema

# High quality domain QA entries spanning agriculture, government schemes, and rural health in EN and HI
SEED_DOMAIN_ENTRIES = [
    # Document 1: PM-KISAN Scheme
    {
        "id": "pmk_en_01",
        "document_id": "doc_pm_kisan_guidelines_2024",
        "domain": "government_schemes",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are a public service advisor on Indian government schemes."},
            {"role": "user", "content": "What is the total annual payment and installment schedule under the PM-KISAN scheme?"},
            {"role": "assistant", "content": "Under the PM-KISAN scheme, eligible beneficiary farmer families receive an annual financial support of Rs. 6,000. This is released directly into their Aadhaar-seeded bank accounts in three equal 4-monthly installments of Rs. 2,000 each via DBT."}
        ]
    },
    {
        "id": "pmk_en_02",
        "document_id": "doc_pm_kisan_guidelines_2024",
        "domain": "government_schemes",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are a public service advisor on Indian government schemes."},
            {"role": "user", "content": "Which categories of farmers are explicitly excluded from PM-KISAN benefits?"},
            {"role": "assistant", "content": "Exclusion criteria under PM-KISAN include: institutional landholders, farmer families holding constitutional posts, former and present Ministers, MPs, MLAs, mayors, chairpersons of district panchayats, serving or retired government employees, pensioners receiving Rs. 10,000 or more per month, and income tax payees from the previous assessment year."}
        ]
    },
    {
        "id": "pmk_hi_01",
        "document_id": "doc_pm_kisan_guidelines_2024",
        "domain": "government_schemes",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप सरकारी योजनाओं और किसान कल्याण पर एक विशेषज्ञ सलाहकार हैं।"},
            {"role": "user", "content": "पीएम किसान सम्मान निधि योजना के तहत कितनी राशि और किन किश्तों में दी जाती है?"},
            {"role": "assistant", "content": "पीएम-किसान योजना के अंतर्गत पात्र किसान परिवारों को प्रति वर्ष ₹6,000 की वित्तीय सहायता दी जाती है। यह राशि ₹2,000 की तीन समान किश्तों में हर चार महीने पर सीधे बैंक खाते में प्रत्यक्ष लाभ अंतरण (DBT) के माध्यम से ट्रांसफर की जाती है।"}
        ]
    },
    {
        "id": "pmk_hi_02",
        "document_id": "doc_pm_kisan_guidelines_2024",
        "domain": "government_schemes",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप सरकारी योजनाओं और किसान कल्याण पर एक विशेषज्ञ सलाहकार हैं।"},
            {"role": "user", "content": "पीएम किसान सम्मान निधि योजना के लिए कौन से किसान अपात्र (excluded) माने गए हैं?"},
            {"role": "assistant", "content": "संस्थागत भूमि धारक, संवैधानिक पदों पर आसीन व्यक्ति, पूर्व या वर्तमान मंत्री, सांसद, विधायक, नगर निगम महापौर, सरकारी कर्मचारी (चतुर्थ श्रेणी को छोड़कर), ₹10,000 से अधिक मासिक पेंशन पाने वाले पेंशनभोगी और पिछले वर्ष आयकर भरने वाले किसान पीएम-किसान के लिए अपात्र हैं।"}
        ]
    },
    # Document 2: PM Fasal Bima Yojana (PMFBY)
    {
        "id": "fby_en_01",
        "document_id": "doc_pmfby_crop_insurance_2024",
        "domain": "government_schemes",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are an agricultural insurance specialist."},
            {"role": "user", "content": "What are the farmer premium share rates for Kharif and Rabi crops under PMFBY?"},
            {"role": "assistant", "content": "Under the Pradhan Mantri Fasal Bima Yojana (PMFBY), farmers pay a maximum premium of 2.0% of the sum insured for Kharif food and oilseed crops, 1.5% for Rabi crops, and 5.0% for annual commercial or horticultural crops. The balance premium is subsidized equally by Central and State Governments."}
        ]
    },
    {
        "id": "fby_hi_01",
        "document_id": "doc_pmfby_crop_insurance_2024",
        "domain": "government_schemes",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप फसल बीमा योजना के आधिकारिक सलाहकार हैं।"},
            {"role": "user", "content": "प्रधानमंत्री फसल बीमा योजना (PMFBY) के तहत खरीफ और रबी फसलों के लिए किसानों को कितना प्रीमियम देना होता है?"},
            {"role": "assistant", "content": "प्रधानमंत्री फसल बीमा योजना के तहत किसानों द्वारा देय अधिकतम प्रीमियम दर खरीफ फसलों के लिए 2%, रबी फसलों के लिए 1.5% तथा वाणिज्यिक एवं बागवानी फसलों के लिए 5% तय की गई है। शेष प्रीमियम का भुगतान केंद्र और राज्य सरकार मिलकर सब्सिडी के रूप में करती हैं।"}
        ]
    },
    # Document 3: Wheat Rust Management
    {
        "id": "wht_en_01",
        "document_id": "doc_wheat_disease_management_icar",
        "domain": "agriculture",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are an ICAR-certified agronomy and crop pathology specialist."},
            {"role": "user", "content": "What are the characteristic field symptoms of stripe/yellow rust in wheat and the recommended chemical spray?"},
            {"role": "assistant", "content": "Yellow (stripe) rust (Puccinia striiformis) presents as bright yellow, powdery pustules arranged in parallel lines or stripes along the leaf veins. For chemical management, spray Propiconazole 25% EC (Tilt) at 0.1% (1 ml per litre of water or 200 ml in 200 litres of water per acre) as soon as the first pustules appear."}
        ]
    },
    {
        "id": "wht_hi_01",
        "document_id": "doc_wheat_disease_management_icar",
        "domain": "agriculture",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप भारतीय कृषि अनुसंधान परिषद (ICAR) के पादप रोग विशेषज्ञ हैं।"},
            {"role": "user", "content": "गेहूं में पीले रतुए के लक्षण क्या हैं और इसके नियंत्रण के लिए किस दवा का छिड़काव करें?"},
            {"role": "assistant", "content": "पीला रतुआ (Yellow Rust) में पत्तियों पर पीले रंग के पाउडर वाले फफोले समानांतर धारियों में दिखाई देते हैं। उंगलियों से छूने पर पीला चूर्ण लग जाता है। इसके नियंत्रण के लिए प्रोपिकोनाजोल 25% ई.सी. (टिल्ट) की 1 मिलीलीटर प्रति लीटर पानी (200 मिली दवा प्रति 200 लीटर पानी प्रति एकड़) की दर से तुरंत छिड़काव करें।"}
        ]
    },
    # Document 4: Cotton Pink Bollworm
    {
        "id": "ctn_en_01",
        "document_id": "doc_cotton_ipm_cicr",
        "domain": "agriculture",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are a crop protection advisor specializing in cotton pest management."},
            {"role": "user", "content": "What is the economic threshold level (ETL) and IPM strategy for pink bollworm in cotton?"},
            {"role": "assistant", "content": "The economic threshold level (ETL) for pink bollworm (Pectinophora gossypiella) is 8 moths per trap per night for 3 consecutive days, or 10% damaged green bolls/rosetted flowers. Management includes deploying pheromone traps at 5 traps/acre, releasing Trichogramma bactrae at 60,000 eggs/acre, and rotating insecticides with profenofos 50% EC or spinosad 45% SC."}
        ]
    },
    {
        "id": "ctn_hi_01",
        "document_id": "doc_cotton_ipm_cicr",
        "domain": "agriculture",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप कपास की फसल में कीट नियंत्रण सलाहकार हैं।"},
            {"role": "user", "content": "कपास में गुलाबी सुंडी (Pink Bollworm) की पहचान और इसके प्रबंधन के मुख्य उपाय बताएं।"},
            {"role": "assistant", "content": "गुलाबी सुंडी के प्रमुख लक्षण फूलों का बंद होकर गुलाब के फूल जैसा (rosetted) दिखना और गोलकों में छेद होना है। नियंत्रण हेतु प्रति एकड़ 5 फेरोमोन ट्रैप लगाएं। आर्थिक क्षति सीमा (8 पतंगे प्रति ट्रैप लगातार 3 दिन) पार होने पर प्रोफेनोफॉस 50% ई.सी. (2 मिली/लीटर) या स्पिनोसैड 45% एस.सी. का छिड़काव करें।"}
        ]
    },
    # Document 5: Soil Health Card Scheme
    {
        "id": "shc_en_01",
        "document_id": "doc_soil_health_card_operational_manual",
        "domain": "agriculture",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are a soil scientist and agricultural extension officer."},
            {"role": "user", "content": "Which 12 parameters are tested and reported on the Soil Health Card?"},
            {"role": "assistant", "content": "The Soil Health Card records 12 parameters: 3 macro-nutrients (Nitrogen [N], Phosphorus [P], Potassium [K]); 1 secondary nutrient (Sulphur [S]); 5 micro-nutrients (Zinc [Zn], Iron [Fe], Copper [Cu], Manganese [Mn], Boron [B]); and 3 physical parameters (pH, Electrical Conductivity [EC], and Organic Carbon [OC])."}
        ]
    },
    {
        "id": "shc_hi_01",
        "document_id": "doc_soil_health_card_operational_manual",
        "domain": "agriculture",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप मृदा स्वास्थ्य विशेषज्ञ हैं।"},
            {"role": "user", "content": "मृदा स्वास्थ्य कार्ड (Soil Health Card) में किन 12 मुख्य मापदंडों की जांच की जाती है?"},
            {"role": "assistant", "content": "मृदा स्वास्थ्य कार्ड में 12 मापदंडों की जांच होती है: मुख्य पोषक तत्व - नाइट्रोजन (N), फास्फोरस (P), पोटाश (K); द्वितीयक पोषक तत्व - सल्फर (S); सूक्ष्म पोषक तत्व - जिंक (Zn), आयरन (Fe), कॉपर (Cu), मैंगनीज (Mn), बोरॉन (B); तथा भौतिक गुण - पीएच मान (pH), विद्युत चालकता (EC) और जैविक कार्बन (OC)।"}
        ]
    },
    # Document 6: Ayushman Bharat PM-JAY
    {
        "id": "jay_en_01",
        "document_id": "doc_pm_jay_operational_guidelines",
        "domain": "rural_health",
        "language": "en",
        "messages": [
            {"role": "system", "content": "You are a healthcare policy specialist for public health schemes."},
            {"role": "user", "content": "What is the financial coverage provided per family under Ayushman Bharat PM-JAY and who qualifies?"},
            {"role": "assistant", "content": "Ayushman Bharat PM-JAY provides health coverage of up to Rs. 5 lakh per family per year for secondary and tertiary care hospitalisation. It covers the bottom 40% of poor and vulnerable families identified via Socio-Economic Caste Census (SECC 2011) criteria, with no restriction on family size, age, or gender, offering cashless hospitalisation across empaneled public and private hospitals."}
        ]
    },
    {
        "id": "jay_hi_01",
        "document_id": "doc_pm_jay_operational_guidelines",
        "domain": "rural_health",
        "language": "hi",
        "messages": [
            {"role": "system", "content": "आप आयुष्मान भारत स्वास्थ्य योजना के अधिकृत परामर्शदाता हैं।"},
            {"role": "user", "content": "आयुष्मान भारत (PM-JAY) योजना में प्रति परिवार कितनी स्वास्थ्य सुरक्षा मिलती है और इसके मुख्य लाभ क्या हैं?"},
            {"role": "assistant", "content": "आयुष्मान भारत योजना के तहत प्रत्येक पात्र परिवार को प्रति वर्ष ₹5 लाख तक का कैशलेस स्वास्थ्य बीमा मिलता है। यह द्वितीयक और तृतीयक देखभाल के लिए देश भर के सूचीबद्ध सरकारी व निजी अस्पतालों में मान्य है। इसमें अस्पताल में भर्ती होने से 3 दिन पहले और 15 दिन बाद तक की दवाओं व जांचों का खर्च भी शामिल होता है।"}
        ]
    }
]


def generate_and_split():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "domain_qa_bilingual.jsonl"

    print(f"Writing {len(SEED_DOMAIN_ENTRIES)} raw domain entries to {raw_file}...")
    with open(raw_file, "w", encoding="utf-8") as f:
        for entry in SEED_DOMAIN_ENTRIES:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Validate
    for idx, entry in enumerate(SEED_DOMAIN_ENTRIES):
        validate_chat_schema(entry, idx)

    # Deduplicate
    unique, removed = deduplicate_dataset(SEED_DOMAIN_ENTRIES)
    print(f"Deduplication complete: {len(unique)} unique entries ({removed} duplicates removed).")

    # Document-level leak-free split (Train: 70%, Val: 15%, Test: 15% for sample)
    train, val, test = split_dataset(
        unique,
        train_ratio=0.67,
        val_ratio=0.17,
        test_ratio=0.16,
        group_key="document_id",
        seed=42,
    )

    paths = save_splits_to_disk(train, val, test, output_dir="data/splits")
    print(f"Splits generated successfully with zero document leakage:")
    print(f"  Train: {len(train)} records -> {paths['train']}")
    print(f"  Val:   {len(val)} records -> {paths['val']}")
    print(f"  Test:  {len(test)} records -> {paths['test']}")


if __name__ == "__main__":
    generate_and_split()
