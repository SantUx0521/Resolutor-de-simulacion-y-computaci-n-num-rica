# De aqui a abajo podemos ir agregando funciones para el proyecto, los temás que tenemos son: Teoremas de taylor, (el resto vamos viendo porque no me acuerdo cuales son XD)
def main():
    print("""
        Bienvenido!! 
        Digite el tema que desea realizar.
            1- Teoremas de taylor
            2- ...
            3- ...
""")
    choosen = int(input("Opcion: "))
    if choosen == 1:
        print("Bienvenido al resolutor de teoremas de taylor.")


if __name__ == "__main__":
    main()