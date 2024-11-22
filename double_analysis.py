import sys
import random
from copy import deepcopy
import numpy as np

# Atom data looks like this


# Atoms # full

# 1645 0 2 -1.7354 -2.789234882470718 -4.745039593134502 -4.880656620688051 2 0 2
# 48 0 1 1.7354 -2.735022446554889 -4.846913774857522 -4.644352406089501 2 0 2

# molecule-id molecule-tag atom-type q x y z nx ny nz

# Bond data looks like this

# Bonds

# 1 1 48 1645
# 2 1 206 1088
# 3 1 468 972
# 4 1 213 1712
# 5 1 475 1788
# 6 1 141 1456
# 7 1 61 1322

# Where each line is a bond between a drude particle and drude core
# bond-id bond-tag drude-particle-id drude-core-id

# I want to be able to parse Bond Data to determine the drude oscillators
# I then want to be able to calculate the total induced dipole of each molecule as a result of the induced dipole
# q (drude core r vector - drude particle vector)

# Following code from DRUDE's polarizer.py, which is also then from pizza.py


# keywords of header and main sections (from data.py in Pizza.py)

hkeywords = [
    "atoms",
    "ellipsoids",
    "lines",
    "triangles",
    "bodies",
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "atom types",
    "bond types",
    "angle types",
    "dihedral types",
    "improper types",
    "xlo xhi",
    "ylo yhi",
    "zlo zhi",
    "xy xz yz",
]

skeywords = [
    ["Masses", "atom types"],
    ["Pair Coeffs", "atom types"],
    ["Bond Coeffs", "bond types"],
    ["Angle Coeffs", "angle types"],
    ["Dihedral Coeffs", "dihedral types"],
    ["Improper Coeffs", "improper types"],
    ["BondBond Coeffs", "angle types"],
    ["BondAngle Coeffs", "angle types"],
    ["MiddleBondTorsion Coeffs", "dihedral types"],
    ["EndBondTorsion Coeffs", "dihedral types"],
    ["AngleTorsion Coeffs", "dihedral types"],
    ["AngleAngleTorsion Coeffs", "dihedral types"],
    ["BondBond13 Coeffs", "dihedral types"],
    ["AngleAngle Coeffs", "improper types"],
    ["Atoms", "atoms"],
    ["Velocities", "atoms"],
    ["Ellipsoids", "ellipsoids"],
    ["Lines", "lines"],
    ["Triangles", "triangles"],
    ["Bodies", "bodies"],
    ["Bonds", "bonds"],
    ["Angles", "angles"],
    ["Dihedrals", "dihedrals"],
    ["Impropers", "impropers"],
    ["Molecules", "atoms"],
]


def massline(att):
    return "{0:4d} {1:8.3f}  # {2}\n".format(att["id"], att["m"], att["type"])


def bdtline(bdt):
    return "{0:4d} {1:12.6f} {2:12.6f}  {3}\n".format(
        bdt["id"], bdt["k"], bdt["r0"], bdt["note"]
    )


def atomline(at):
    return (
        "{0:7d} {1:7d} {2:4d} {3:8.4f} {4:13.6e} {5:13.6e} {6:13.6e} "
        " {7}\n".format(
            at["n"], at["mol"], at["id"], at["q"], at["x"], at["y"], at["z"], at["note"]
        )
    )


def bondline(bd):
    return "{0:7d} {1:4d} {2:7d} {3:7d}  {4}\n".format(
        bd["n"], bd["id"], bd["i"], bd["j"], bd["note"]
    )


def velline(at):
    return "{0:7d} {1:13.6e} {2:13.6e} {3:13.6e} \n".format(
        at["n"], at["vx"], at["vy"], at["vz"]
    )


# --------------------------------------


class Data(object):

    def __init__(self, datafile):
        """read LAMMPS data file (from data.py in Pizza.py)"""

        # for extract method
        self.atomtypes = []
        self.bondtypes = []
        self.atoms = []
        self.bonds = []
        self.idmap = {}

        self.nselect = 1

        f = open(datafile, "r")

        self.title = f.readline()
        self.names = {}

        headers = {}
        while 1:
            line = f.readline().strip()
            if "#" in line:
                line = line[: line.index("#")].strip()
            if len(line) == 0:
                continue
            found = 0
            for keyword in hkeywords:
                if keyword in line:
                    found = 1
                    words = line.split()
                    if (
                        keyword == "xlo xhi"
                        or keyword == "ylo yhi"
                        or keyword == "zlo zhi"
                    ):
                        headers[keyword] = (float(words[0]), float(words[1]))
                    elif keyword == "xy xz yz":
                        headers[keyword] = (
                            float(words[0]),
                            float(words[1]),
                            float(words[2]),
                        )
                    else:
                        headers[keyword] = int(words[0])
            if not found:
                break

        sections = {}
        while 1:
            if len(line) > 0:
                found = 0
                for pair in skeywords:
                    keyword, length = pair[0], pair[1]
                    if keyword == line:
                        found = 1
                        if length not in headers:
                            raise RuntimeError(
                                "data section {} "
                                "has no matching header value".format(line)
                            )
                        f.readline()
                        list_ = []
                        for _ in range(headers[length]):
                            list_.append(f.readline())
                        sections[keyword] = list_
                if not found:
                    raise RuntimeError(
                        "invalid section {} in data" " file".format(line)
                    )
            # f.readline()
            line = f.readline()
            if not line:
                break
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()

        f.close()
        self.headers = headers
        self.sections = sections

    def extract_pol(self):
        """extract atom, drude, bonds info from polarizable data"""

        # extract atom IDs
        for line in self.sections["Masses"]:
            tok = line.split()
            atomtype = {}
            atomtype["id"] = int(tok[0])
            atomtype["m"] = float(tok[1])
            if len(tok) >= 4:
                atomtype["type"] = tok[3]
                atomtype["dflag"] = "n"
                if tok[-1] == "DC":
                    atomtype["dflag"] = "c"
                elif tok[-1] == "DP":
                    atomtype["dflag"] = "d"
                print(atomtype["dflag"])
            else:
                raise RuntimeError(
                    "comments in Masses section required "
                    "to identify cores (DC) and Drudes (DP)"
                )
            self.atomtypes.append(atomtype)

        # extract bond type data
        for line in self.sections["Bond Coeffs"]:
            tok = line.split()
            bondtype = {}
            bondtype["id"] = int(tok[0])
            bondtype["k"] = float(tok[1])
            bondtype["r0"] = float(tok[2])
            bondtype["note"] = "".join([s + " " for s in tok[3:]]).strip()
            self.bondtypes.append(bondtype)

        # extract atom registers
        for line in self.sections["Atoms"]:
            tok = line.split()
            atom = {}
            atom["n"] = int(tok[0])
            atom["mol"] = int(tok[1])
            atom["id"] = int(tok[2])
            atom["q"] = float(tok[3])
            atom["x"] = float(tok[4])
            atom["y"] = float(tok[5])
            atom["z"] = float(tok[6])
            atom["xu"] = int(tok[7])
            atom["yu"] = int(tok[8])
            atom["zu"] = int(tok[9])
            # atom['note'] = ''.join([s + ' ' for s in tok[7:-1]]).strip()
            if tok[-1] == "DC":
                atom["note"] = " ".join(tok[7:-1])
            else:
                atom["note"] = " ".join(tok[7:])
            self.atoms.append(atom)
            self.idmap[atom["n"]] = atom

        if "Velocities" in self.sections:
            for line in self.sections["Velocities"]:
                tok = line.split()
                atom = self.idmap[int(tok[0])]
                atom["vx"] = float(tok[1])
                atom["vy"] = float(tok[2])
                atom["vz"] = float(tok[3])

        # extract bond data
        for line in self.sections["Bonds"]:
            tok = line.split()
            bond = {}
            bond["n"] = int(tok[0])
            bond["id"] = int(tok[1])
            bond["i"] = int(tok[2])
            bond["j"] = int(tok[3])
            bond["note"] = "".join([s + " " for s in tok[4:]]).strip()
            self.bonds.append(bond)


# --------------------------------------

kcal = 4.184  # kJ
eV = 96.485  # kJ/mol
fpe0 = 0.000719756  # (4 Pi eps0) in e^2/(kJ/mol A)

import numpy as np
import matplotlib.pyplot as plt

def string_to_atom(line):
    # Example string
    # 581 0 1 C 1.7354 -5.05554 -4.70332 -4.74188
    words = line.split()
    return Atom(int(words[0]), int(words[1]), int(words[2]), words[3], float(words[4]), float(words[5]), float(words[6]), float(words[7]))

class Atom(object):
    # ATOMS id mol type element q xu yu zu
    def __init__(self, id, mol, type, element, q, x, y, z):
        self.id = id
        self.mol = mol
        self.type = type
        self.element = element
        self.q = q
        self.x = x
        self.y = y
        self.z = z

    def dipole(self):
        return self.q * np.array([self.x, self.y, self.z])

class Timestep(object):
    def __init__(self, timestep, atoms):
        self.timestep = timestep
        self.atoms = atoms
        self.dipole_moment, self.dipole_magnitude = self.dipole()

    def dipole(self):
        dipoles = np.array([atom.dipole() for atom in self.atoms])
        dipole_moment = np.sum(dipoles, axis=0)
        # return sum of dipoles and magnitude of sum
        return dipole_moment, np.linalg.norm(dipole_moment)

class Dump(object):
    def __init__(self, datafile):
        """Read LAMMPS dump file"""
        self.timesteps = []
        self.datafile = datafile
        self.read_dump()

    def read_dump(self):
        # Dump file looks like this
        # ITEM: TIMESTEP
        # 100
        # ITEM: NUMBER OF ATOMS
        # 1800
        # ITEM: BOX BOUNDS pp pp pp
        # -5.5000000000000000e+00 5.5000000000000000e+00
        # -5.5000000000000000e+00 5.5000000000000000e+00
        # -5.5000000000000000e+00 5.5000000000000000e+00
        # ITEM: ATOMS id mol type element q xu yu zu
        # 581 0 1 C 1.7354 -5.05554 -4.70332 -4.74188
        # 716 0 1 C 1.7354 -4.34815 -4.10895 -4.23278
        # 91 0 1 C 1.7354 -3.81375 -3.53894 -3.51429
        # ... more atoms
        # repeat for each timestep
        with open(self.datafile, 'r') as f:
            while True:
                # Read timestep
                line = f.readline()
                if not line:
                    break  # End of file
                if "ITEM: TIMESTEP" not in line:
                    continue
                
                timestep = int(f.readline().strip())
                
                # Skip "ITEM: NUMBER OF ATOMS" line
                f.readline()
                num_atoms = int(f.readline().strip())
                
                # Skip "ITEM: BOX BOUNDS" and the three lines after it
                for _ in range(4):
                    f.readline()

                # Skip "ITEM: ATOMS" line
                f.readline()
                
                # Read atoms
                atoms = []
                for _ in range(num_atoms):
                    line = f.readline().strip()
                    atom = string_to_atom(line)
                    atoms.append(atom)
                
                # Create Timestep object and add to list
                self.timesteps.append(Timestep(timestep, atoms))


def main():
    # open cool_stuff.data
    data = Data("cooler_stuff.data")
    # list the bonds
    data.extract_pol()
    total_dipole = [0, 0, 0]

    # Dictionary to store molecules
    molecules = {}

    # Group atoms into molecules based on bonds
    for bond in data.bonds:
        mol_id = bond["n"]
        if mol_id not in molecules:
            molecules[mol_id] = set()
        molecules[mol_id].add(bond["i"])
        molecules[mol_id].add(bond["j"])

    dump = Dump('dump-two.lammpstrj')
    # write data to a file
    with open('dipole_magnitude_vs_time.txt', 'w') as f:
        for ts in dump.timesteps:
            f.write(f"{ts.timestep} {ts.dipole_moment} {ts.dipole_magnitude}\n")
    # graph dipole magnitude vs time using matplotlib
    # Extract timesteps and dipole magnitudes
    timesteps = [ts.timestep for ts in dump.timesteps]
    dipole_magnitudes = [ts.dipole_magnitude for ts in dump.timesteps]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, dipole_magnitudes, '-o')
    plt.title('Dipole Magnitude vs Time')
    plt.xlabel('Timestep')
    plt.ylabel('Dipole Magnitude')
    plt.grid(True)

    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('dipole_magnitude_vs_time.png')

    print(timesteps, dipole_magnitudes)
    
    # Display the plot (comment this out if running on a server without display)
    plt.show()

    # Print some statistics
    print(f"Average dipole magnitude: {np.mean(dipole_magnitudes):.4f}")
    print(f"Maximum dipole magnitude: {np.max(dipole_magnitudes):.4f}")
    print(f"Minimum dipole magnitude: {np.min(dipole_magnitudes):.4f}")


if __name__ == '__main__':
    main()
