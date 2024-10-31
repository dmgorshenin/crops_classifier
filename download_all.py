from crops_classifier import AgroClassifierService


def main():
    ac = AgroClassifierService('./configs/astrakhan.json', 'y')
    try:
        ac.classify_data()
    except Exception as e:
        raise e
    
    ac = AgroClassifierService('./configs/samara.json', 'y')
    try:
        ac.classify_data()
    except Exception as e:
        raise e

    ac = AgroClassifierService('./configs/saratov.json', 'y')
    try:
        ac.classify_data()
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()