def main():
    print('''WELCOME TO THE HMM Part of Speech Tagger

        Enter the number in order to:
            (1) Train polish model 
            (2) Train portuguese model
            (3) Exit

        By Ane García, Marcos Merino and Julia Wojciechowska\n''')

    eleccion = input()

    if int(eleccion) == 1:
        print("Training polish model")
        main()

    elif int(eleccion) == 2:
        print("Training portuguese model")
        main()

    elif int(eleccion) == 4:
        print("EXIT...")
        return

    else:
        print("Incorrect selection\n\n")
        main()


if __name__ == "__main__":
    main()