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
